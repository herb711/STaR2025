import math
import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BitsAndBytesConfig
import igraph as ig
from collections import defaultdict

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class BasicGenerator:
    def __init__(self, model_name_or_path):
        # 使用 4bit 量化加载模型，节省内存和计算资源
        logger.info(f"Loading model with 4bit quantization from {model_name_or_path}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                
            bnb_4bit_quant_type="nf4",        
            bnb_4bit_use_double_quant=True,   
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code="falcon" in model_name_or_path,
            attn_implementation="eager",
            quantization_config=quantization_config,     
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)

        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text, max_length, return_logprobs=False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        temperature = 0.3
        top_p = 1.0
        top_k = 50
        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,                
                pad_token_id = self.tokenizer.eos_token_id,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
                pad_token_id = self.tokenizer.eos_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0])
            return text, None, None

    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        temperature = 0.3
        top_p = 1.0
        top_k = 50

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            pad_token_id = self.tokenizer.eos_token_id,
            max_new_tokens = max_length,
            return_dict_in_generate = True,
            output_scores = True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated_tokens = outputs.sequences[:, input_length:]

        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])

        text = self.tokenizer.decode(generated_tokens[0])

        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError(f"Solver '{solver}' not implemented.")
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            sum_atten = sum(mean_atten[1:]).item()
            if sum_atten > 1e-9:
                 mean_atten = mean_atten / sum_atten
            else:
                 pass

        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(np.array(tmp, dtype=np.float64), axis=-1)
            log_probs = np.log(softmax_probs + 1e-10)
            product = softmax_probs * log_probs
            product = np.nan_to_num(product, nan=0.0)
            entropies = -np.sum(product, axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None

        return text, seqlist, attns, seqlogprobs, seqentropies

    def compute_word_centrality(self, text, window_size=2, return_word_text=False):
        effective_max_length = self.tokenizer.model_max_length
        reasonable_upper_bound = 4096
        if effective_max_length > reasonable_upper_bound * 10:
            effective_max_length = reasonable_upper_bound

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=effective_max_length
        )
        offsets = encoding["offset_mapping"][0].tolist()
        input_ids = encoding["input_ids"][0]

        num_tokens = len(input_ids)
        if num_tokens == 0:
             return ({}, {}) if return_word_text else ({}, None)

        G = ig.Graph(directed=False)
        G.add_vertices(num_tokens)
        edges = []
        for i in range(num_tokens):
            for j in range(i + 1, min(i + window_size, num_tokens)):
                 edges.append((i, j))
        if edges:
            G.add_edges(edges)

        if G.vcount() > 0 and G.ecount() > 0:
            token_centrality = G.pagerank()
        elif G.vcount() > 0:
            token_centrality = [0.0] * G.vcount()
        else:
            token_centrality = []

        span_scores = defaultdict(list)
        span_to_word_text = {}

        for i, (start, end) in enumerate(offsets):
            if start == end or (start == 0 and end == 0 and i > 0):
                continue
            if i < len(token_centrality):
                span = (start, end)
                span_scores[span].append(token_centrality[i])
                if return_word_text:
                    if start < len(text) and end <= len(text):
                         span_to_word_text[span] = text[start:end]
                    else:
                         logger.warning(f"Offset span {span} out of bounds for text length {len(text)}. Token index: {i}")

        word_centrality = {
            span: sum(scores) / len(scores) if scores else 0.0
            for span, scores in span_scores.items()
        }

        if return_word_text:
            return word_centrality, span_to_word_text
        else:
            return word_centrality, None

    def compute_token_centralities(self, text, target_tokens):
        word_centrality, span_to_word = self.compute_word_centrality(
            text, return_word_text=True)

        def normalize_word_centrality(word_centrality: dict, power: float = 1.0) -> dict:
            if not word_centrality:
                return {}

            values = list(word_centrality.values())
            min_val, max_val = min(values), max(values)
            eps = 1e-9

            if abs(max_val - min_val) < eps:
                return {k: 0.0 for k in word_centrality}

            return {
                k: ((v - min_val) / (max_val - min_val + eps)) ** power
                for k, v in word_centrality.items()
            }

        word_centrality = normalize_word_centrality(word_centrality, power=1.5)

        if any(np.isnan(v) for v in word_centrality.values()):
            logger.warning("NaN detected in word_centrality values!")

        centralities = []
        token_pos = 0
        span_map = list(span_to_word.items())

        for tok in target_tokens:
            matched = False
            for (start, end), word in span_map:
                if token_pos >= start and token_pos < end:
                    if (start, end) in word_centrality:
                        centralities.append(word_centrality[(start, end)])
                        matched = True
                        break
            if not matched:
                centralities.append(0.0)
            token_pos += 1

        return centralities


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text

class STaR_v6(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        
        self.halluc_gamma = getattr(args, 'halluc_gamma', 0.3)

        self.centrality_weight = getattr(args, 'centrality_weight', 0.1)

        self.consistency_penalty = getattr(args, 'consistency_penalty', 5.0)

        self.halluc_f1 = getattr(args, 'halluc_f1', True)        

        self.query_harmonic = getattr(args, 'query_harmonic', True)     

        self.dynamic_halluc_threshold_open = getattr(args, 'dynamic_halluc_threshold_open', True)

    def find_token_index_of_phrase(self, text, tokens, phrase, tokenizer, start=0):
        lower_text = text.lower()
        phrase = phrase.lower()
        pos = lower_text.find(phrase, start)
        if pos == -1:
            return -1
        token_prefix = tokenizer.encode(text[:pos + len(phrase)], add_special_tokens=False)
        return len(token_prefix) - 1

    def modifier(self, text, tokens, attentions, weight, iteration):
        ans_pos = text.lower().find("the answer is")
        question_token_idx = None
        if ans_pos != -1:
            question_pos = text.lower().find("question:", ans_pos)
            if question_pos != -1:
                question_token_idx = self.find_token_index_of_phrase(
                    text, tokens, "question:", tokenizer=self.generator.tokenizer, start=ans_pos
                )
                
        centralities = self.generator.compute_token_centralities(text, tokens)
        
        eps = 1e-9

        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr

            attns = attentions[tl:tr]
            attns = np.array(attns) / (sum(attns) + eps)
            length_factor = (tr - tl)
            halluc_scores = []
            for i in range(tl, tr):
                att = attns[i - tl]
                w = weight[i]   
                cent = centralities[i] if i < len(centralities) else 0.0

                UA = att * w * length_factor 
                structural_edge = 1.0 - cent * self.centrality_weight 

                if self.halluc_f1 == True:
                    score = (UA * structural_edge) / (UA + structural_edge + eps)
                else:
                    score = UA * structural_edge

                halluc_scores.append(score)            

            logger.debug(f"All hallucination scores: {halluc_scores}")
       
            b = 0.5            

            if self.dynamic_halluc_threshold_open:     
                b = 0.5
                hallucination_threshold = min(self.hallucination_threshold * math.log2(iteration + b), 2.0)                
            else:
                hallucination_threshold = self.hallucination_threshold                

            thres = [1 if v > hallucination_threshold else 0 for v in halluc_scores]

            if 1 in thres and question_token_idx is not None: 
                
                if tl >= question_token_idx:                
                    thres = [0] * (tr - tl)                    
                elif tr > question_token_idx:                    
                    for i in range(max(question_token_idx, tl), tr):
                        thres[i - tl] = 0
                    
            if 1 in thres:
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0
                    if 1 not in thres:                         
                         continue

                prev = "" if sid == 0 else " ".join(sentences[:sid])
                                
                return True, prev, tokens[tl:tr], thres 

        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        logger.debug("Keeping real words for retrieval query formulation.")
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt").to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        with torch.no_grad():
             atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            att = torch.zeros(input_length, device=input_ids.device)
            for i in range(r[0], r[1] + 1):
                if i == 0: continue
                v = atten_tmp[i-1][:r[0]]
                if v.sum() > 1e-9:
                    v = v / v.sum()
                else:
                    v = torch.zeros_like(v)
                t = torch.zeros(input_length, device=input_ids.device)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            merged_att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_], device=input_ids.device)
            attns.append(merged_att)

        forward_attns = torch.zeros(len(tokens), device=input_ids.device)
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        if hit_cnt > 0:
            forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        centralities = self.generator.compute_token_centralities(all_text, tokens)

        epsilon = 1e-9
        real_pairs = []
        for i in range(len(tokens)):            
            if i >= curr_st and curr_hit[i - curr_st]:
                continue
            tok = tokens[i]
            if match(tok):
                if self.query_harmonic == True:
                    att = forward_attns[i]
                    cent = centralities[i] if i < len(centralities) else 0.0
                    cent = cent ** self.halluc_gamma

                    att = max(att, 1e-5)

                    harmonic = (2 * att * cent) / (att + cent + epsilon)
                    delta = abs(att - cent)
                    consistency = 1.0 / (1 + self.consistency_penalty * delta)

                    score = harmonic * consistency
                else:
                    score = float(att)
                
                if score > 0.001:
                    real_pairs.append((score, tok, i))
                
                logger.debug(f"token='{tok}' | att={att:.4f} | cent={cent:.4f} | score={score:.4f}")

        query_score_pairs = [(tok, score) for score, tok, _ in real_pairs]
        formatted = " | ".join([f"{tok} -> {score:.4f}" for tok, score in query_score_pairs])
        logger.info(f"Retrieval query from score: {formatted}")

        top_k = len(real_pairs)
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        query = " ".join([x[1] for x in real_pairs])
        
        return query

    def inference(self, question, demo, case):
        logger.info(f"inference for question: '{question}'")
        text = ""
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            old_len = len(text)

            prompt = "".join([d["case"]+"\n" for d in demo])            
            tmp_li = [case, text]
            prompt += " ".join(s for s in tmp_li if len(s) > 0)
            logger.debug(f"Generation prompt (Iteration {iteration}):\n{prompt}")

            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_max_length,
                use_entropy = self.method == "STaR",
                use_logprob = self.method == "attn_prob"
            )
            
            if self.method == "STaR" and entropies is not None:
                if np.isnan(entropies).any():
                    logger.warning("NaN detected in raw entropies from generate_attn!")
            elif self.method == "attn_prob" and logprobs is not None:
                if np.isnan(logprobs).any():
                    logger.warning("NaN detected in raw logprobs from generate_attn!")

            base_weight = []
            if self.method == "STaR" and entropies is not None:
                 base_weight = entropies
                 nan_indices = np.isnan(base_weight)
                 if np.any(nan_indices):
                     num_nans = np.sum(nan_indices)
                     logger.warning(f"Found {num_nans} NaN(s) in base_weight (entropies). Replacing with 0.0.")
                     base_weight = np.nan_to_num(base_weight, nan=0.0).tolist()
            elif self.method == "attn_prob" and logprobs is not None:
                 base_weight = [-v for v in logprobs]
                 nan_indices = np.isnan(base_weight)
                 if np.any(nan_indices):
                     num_nans = np.sum(nan_indices)
                     logger.warning(f"Found {num_nans} NaN(s) in base_weight (neg logprobs). Replacing with 0.0.")
                     base_weight = np.nan_to_num(base_weight, nan=0.0).tolist()
            else:
                 logger.warning(f"Could not determine base weight for method '{self.method}'. Using zeros.")
                 base_weight = [0.0] * len(tokens)

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            
            hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, base_weight, iteration)

            if not hallucination:
                text = text.strip() + " " + new_text.strip()                
            else:
                self.counter.hallucinated += 1
                forward_all = [question, text, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = tokenizer.convert_tokens_to_string(last_n_tokens)
                    return last_n_sentence

                retrieve_question = ""
                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)
                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )
                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all
                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__, "retrieve_keep_top_k must be set for last_n_tokens query formulation"
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)
                elif self.query_formulation == "real_words":
                    if iteration == 1:
                        retrieve_question = question
                    else:
                        retrieve_question = self.keep_real_words(
                            prev_text = question + " " + text + " " + ptext,
                            curr_tokens = curr_tokens,
                            curr_hit = curr_hit,
                        )
                        if retrieve_question == "":
                            retrieve_question = question
                else:
                    logger.error(f"Unsupported query formulation: {self.query_formulation}")
                    raise NotImplementedError
                
                logger.info(f"Retrieval query: {retrieve_question}")
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                logger.debug(f"Retrieval successful. Retrieved {len(docs)} documents.")

                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"                
                tmp_li = [case, text]
                prompt += " ".join(s for s in tmp_li if len(s) > 0)                
                logger.info(f"Regeneration prompt with context (Iteration {iteration}):\n{prompt[prompt.find('Context:'):]}")

                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)

                new_text = self.get_top_sentence(new_text)                
                tmp_li = [text.strip(), new_text.strip()]
                text = " ".join(s for s in tmp_li if len(s) > 0)
                
            tokens_count = len(self.generator.tokenizer.encode(text))            
            if "the answer is" in text.lower():
                 logger.info("Stopping criteria met: Found 'the answer is'.")
                 break
            if tokens_count >= self.generate_max_length:
                 logger.info("Stopping criteria met: Max generation length reached.")
                 break
            if len(text) <= old_len:
                 logger.info("Stopping criteria met: Text length did not increase.")
                 break

        return text

# ... rest of the file ...