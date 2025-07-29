import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging # 导入 logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC
from generate import * # 确保使用了最终的类名

# 移除顶层的 basicConfig 和 getLogger
# 配置将在 main 函数中进行

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    # 解析命令行参数以获取 config_path
    cmd_args = parser.parse_args()
    config_path = cmd_args.config_path
    # 从 JSON 文件加载配置
    with open(config_path, "r", encoding='utf-8') as f: # 指定编码
        args_dict = json.load(f)
    # 将字典转换为 Namespace 对象
    args = argparse.Namespace(**args_dict)
    # 保留原始 config_path
    args.config_path = config_path
    # 设置默认值（如果不存在）
    if not hasattr(args, "shuffle"):
        args.shuffle = False
    if not hasattr(args, "use_counter"):
        args.use_counter = True
    if not hasattr(args, "sample"): # 添加 sample 的默认值检查
        args.sample = -1
    # 确保 output_dir 存在
    if not hasattr(args, "output_dir"):
         args.output_dir = "./output_default" # 提供一个默认值
         # 在配置日志前无法使用 logger，暂时用 print
         print(f"Warning: 'output_dir' not found in config, using default '{args.output_dir}'")
    return args

def setup_logging(log_dir):
    """配置日志记录，仅输出到文件""" # 修改文档字符串
    # log_level = logging.DEBUG # 设置级别为 DEBUG
    log_level = logging.INFO # 设置级别为 INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_filename = 'run_experiment.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # --- 修改：移除 StreamHandler ---
    # 手动配置或者在 basicConfig 中移除 StreamHandler
    # 这里选择修改 basicConfig 的 handlers 列表
    log_handlers = []
    file_handler = None
    try:
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        log_handlers.append(file_handler) # 只添加文件处理器
    except Exception as e:
         import sys
         print(f"CRITICAL: Error creating file handler for {log_filepath}: {e}", file=sys.stderr)
         # 如果文件处理器创建失败，可以决定是否继续（无文件日志）或退出

    logging.basicConfig(level=log_level,
                        format=log_format,
                        handlers=log_handlers, # 使用只包含 FileHandler 的列表
                        force=True)
    # --- 结束修改 ---

    # 获取主脚本 logger
    logger = logging.getLogger(__name__) # 获取 __main__ 的 logger
    logger.info(f"--- Logging Initialized (File Only) ---")
    logger.info(f"Log Level: {logging.getLevelName(log_level)}")
    if file_handler: # 检查文件处理器是否成功创建
        logger.info(f"Log File: {log_filepath}")
    else:
        logger.warning(f"File logging disabled due to handler creation error.")
    logger.info(f"------------------------------------")
    return logger # 返回主脚本的 logger 实例

def main():
    args = get_args()

    # --- 确定并创建最终输出目录 ---
    output_base_dir = args.output_dir
    final_output_dir = None # 初始化
    try:
        if not os.path.exists(output_base_dir):
            # 使用 print 因为日志还未配置
            print(f"Creating base output directory: {output_base_dir}")
            os.makedirs(output_base_dir)
        # 寻找下一个可用的子目录编号
        for i in range(10000):
            potential_dir = os.path.join(output_base_dir, str(i))
            if not os.path.exists(potential_dir):
                final_output_dir = potential_dir
                os.makedirs(final_output_dir)
                print(f"Created output subdirectory: {final_output_dir}")
                break
        if final_output_dir is None: # 如果循环结束仍未找到
             final_output_dir = output_base_dir # 回退到基础目录
             print(f"Warning: Could not create a new numbered subdirectory. Using base directory: {final_output_dir}")

    except OSError as e:
        print(f"Error creating output directory structure: {e}. Using current directory '.' as fallback.")
        final_output_dir = "." # 最终回退到当前目录

    args.output_dir = final_output_dir # 更新 args 中的路径为最终路径

    # --- 在确定最终输出目录后配置日志 (现在仅写入文件) ---
    logger = setup_logging(args.output_dir)

    # --- 现在可以使用 logger 记录信息了 ---
    logger.info(f"Final output directory set to: {args.output_dir}")
    # 使用 json.dumps 让 Namespace 打印更美观
    logger.info(f"Running with arguments:\n{json.dumps(args.__dict__, indent=4)}")

    # --- 保存配置文件到最终输出目录 ---
    config_save_path = os.path.join(args.output_dir, "config.json")
    try:
        with open(config_save_path, "w", encoding='utf-8') as f:
            json.dump(args.__dict__, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration saved to {config_save_path}")
    except IOError as e:
        logger.error(f"Failed to save configuration to {config_save_path}: {e}")

    # --- 创建输出文件 ---
    output_file_path = os.path.join(args.output_dir, "output.txt")
    output_file = None # 初始化为 None
    try:
        # 使用 utf-8 编码打开文件
        output_file = open(output_file_path, "w", encoding='utf-8')
        logger.info(f"Output file created at {output_file_path}")

        # --- 加载数据 ---        
        try:
            if args.dataset == "strategyqa":
                data = StrategyQA(args.data_path)
            elif args.dataset == "2wikimultihopqa":
                data = WikiMultiHopQA(args.data_path)
            elif args.dataset == "hotpotqa":
                data = HotpotQA(args.data_path)
            elif args.dataset == "iirc":
                data = IIRC(args.data_path)
            else:
                raise NotImplementedError(f"Dataset '{args.dataset}' not implemented.")
            data.format(fewshot=args.fewshot)
            data = data.dataset
            if args.shuffle:
                logger.info("Shuffling dataset.")
                data = data.shuffle()
            if args.sample != -1:
                samples = min(len(data), args.sample)
                data = data.select(range(samples))
            logger.info(f"Data loaded. Number of examples: {len(data)}")
        except Exception as e:
            logger.error(f"加载或处理数据时出错: {e}", exc_info=True)
            if output_file and not output_file.closed: output_file.close() # 关闭已打开的文件
            return

        # --- 根据 method 选择并初始化模型 ---
        # token(FLARE)、fix-length-retrieval(FL-RAG) 、fix-sentence-retrieval(FS-RAG) 、single-retrieval(SR-RAG) non-retrieval（wo-RAG）
        logger.info(f"Initializing model with method: {args.method}")
        try:
            if args.method == "non-retrieval":
                model = BasicRAG(args)
            elif args.method == "STaR":
                # 确保使用您最终修改的那个类名
                model = STaR_v6(args) # 例如 STaR_v2_Final
            else:
                raise NotImplementedError(f"Method '{args.method}' not implemented.")
            logger.info(f"Model '{model.__class__.__name__}' initialized successfully.")
        except Exception as e:
            logger.error(f"初始化模型时出错: {e}", exc_info=True)
            if output_file and not output_file.closed: output_file.close()
            return


        # --- 开始推理 ---
        logger.info("Starting inference loop...")
        for i in tqdm(range(len(data)), desc="Inference Progress"):
            batch = data[i]
            qid = batch.get("qid", f"item_{i}") # 获取 qid，提供回退值
            logger.info(f"\n--- Processing QID: {qid} ---")
            last_counter = copy(model.counter) # 复制计数器状态
            try:
                pred = model.inference(batch["question"], batch["demo"], batch["case"])
                pred = pred.strip()
                # logger.debug(f"QID {qid} Prediction (first 100 chars): '{pred[:100]}...'") # 记录部分预测结果
            except Exception as e:
                logger.error(f"Error during inference for QID {qid}: {e}", exc_info=True)
                pred = "[Inference Error]" # 记录错误信息而非崩溃

            ret = {
                "qid": qid,
                "prediction": pred,
            }
            if args.use_counter:
                try:
                  ret.update(model.counter.calc(last_counter))
                except Exception as e:
                  logger.error(f"Error calculating counter diff for QID {qid}: {e}")

            # 将结果写入文件，确保每次写入后刷新缓冲区
            try:
                # 确保 output_file 已成功打开
                if output_file and not output_file.closed:
                    output_file.write(json.dumps(ret, ensure_ascii=False)+"\n")
                    output_file.flush() # 强制写入磁盘，便于实时查看
                else:
                    logger.error(f"Output file is not open. Cannot write result for QID {qid}.")
            except Exception as e:
                logger.error(f"Error writing output for QID {qid}: {e}")

    except Exception as e: # 捕获主循环中的其他潜在错误
        logger.error(f"An error occurred during the main inference loop: {e}", exc_info=True)
    finally:
        # 确保文件在程序结束或出错时被关闭
        if output_file is not None and not output_file.closed:
            output_file.close()
            logger.info("Output file closed.")

    logger.info("Inference process finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 捕获 main 函数中的未处理异常
        # 尝试记录到日志（如果配置成功）
        try:
             # 尝试获取 logger，如果失败则无法记录
             logger = logging.getLogger(__name__)
             # 检查 logger 是否有处理器，以防 setup_logging 失败
             if logger.hasHandlers():
                 logger.critical(f"Unhandled critical exception in main: {e}", exc_info=True)
             else: # 如果没有处理器，打印到 stderr
                 import sys
                 print(f"CRITICAL UNHANDLED EXCEPTION in main (logging not configured): {e}", file=sys.stderr)
        except Exception: # 如果日志本身也出错，打印到 stderr
             import sys
             print(f"CRITICAL UNHANDLED EXCEPTION in main (logging failed): {e}", file=sys.stderr)
        exit(1) # 异常退出