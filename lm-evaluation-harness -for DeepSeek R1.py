import os
import logging
import json
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import Dict, Any, List
import tiktoken
from datasets import load_dataset
import re

class RemoteModelEvaluator:
    def __init__(self):
        """初始化评估器"""
        # 首先设置基本属性
        self.api_url = "http://22a91545c9.51mypc.cn:28081/v1/chat/completions"
        self.model_name = "DeepSeek-R1-INT4"
        self.suite_name = f"deepseek-benchmark-{int(time.time())}"
        self.semaphore = asyncio.Semaphore(4)
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.timeout = aiohttp.ClientTimeout(total=300)  # 30 seconds timeout
        self.output_file = None  # Will be set in initialize_results_file

        # 然后设置日志
        self.setup_logging()
        
        # 最后初始化结果文件
        self.initialize_results_file()

    async def get_model_response(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """获取模型响应，带重试机制"""
        for retry in range(self.max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "stream": False
                }

                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    elif response.status == 429:  # Rate limit
                        if retry < self.max_retries - 1:
                            wait_time = self.retry_delay * (retry + 1)
                            logging.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {retry + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                    elif response.status == 503 or response.status == 502:  # Service unavailable
                        if retry < self.max_retries - 1:
                            wait_time = self.retry_delay * (retry + 1)
                            logging.warning(f"Service temporarily unavailable, waiting {wait_time} seconds before retry {retry + 1}")
                            await asyncio.sleep(wait_time)
                            continue
                    else:
                        raise Exception(f"API call failed: {response.status} - {response_text}")

            except asyncio.TimeoutError:
                if retry < self.max_retries - 1:
                    wait_time = self.retry_delay * (retry + 1)
                    logging.warning(f"Request timeout, waiting {wait_time} seconds before retry {retry + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception("Request timeout after all retries")

            except aiohttp.ClientError as e:
                if retry < self.max_retries - 1:
                    wait_time = self.retry_delay * (retry + 1)
                    logging.warning(f"Connection error: {str(e)}, waiting {wait_time} seconds before retry {retry + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Connection error after all retries: {str(e)}")

            except Exception as e:
                if retry < self.max_retries - 1:
                    wait_time = self.retry_delay * (retry + 1)
                    logging.warning(f"Unexpected error: {str(e)}, waiting {wait_time} seconds before retry {retry + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Error in get_model_response after all retries: {str(e)}")

        raise Exception("Failed to get model response after all retries")

    async def create_session(self) -> aiohttp.ClientSession:
        """创建带有重试机制的会话"""
        connector = aiohttp.TCPConnector(
            limit=10,  # 限制并发连接数
            force_close=True,  # 强制关闭连接
            enable_cleanup_closed=True,  # 清理关闭的连接
            ssl=False  # 禁用SSL验证
        )
        return aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={
                "Content-Type": "application/json",
                "Connection": "keep-alive"
            }
        )

    async def evaluate_mmlu(self, subject: str) -> Dict[str, Any]:
        """评估MMLU特定学科的得分"""
        try:
            logging.info(f"Starting evaluation MMLU - {subject}")
            dataset = load_dataset("cais/mmlu", subject, split="test")
            total_samples = len(dataset)
            logging.info(f"Loaded {total_samples} test samples")

            correct = 0
            total = 0
            responses = []

            async with aiohttp.ClientSession() as session:
                for idx, item in enumerate(dataset):
                    question = item["question"]
                    choices = item["choices"]
                    correct_answer = item["answer"]

                    prompt = f"""问题：{question}

选项：
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

请分析每个选项，并在最后用"答案："标记您的选择（仅填写A、B、C或D）。"""

                    try:
                        response = await self.get_model_response(session, prompt)
                        if not response:  # 检查空响应
                            raise Exception("Received empty response from model")

                        # 1. 首先尝试查找"答案："后的内容
                        final_answer_match = re.search(r'答案：\s*([ABCD])', response.upper())

                        if final_answer_match:
                            answer_letter = final_answer_match.group(1)
                        else:
                            # 2. 如果没有找到标记，查找最后出现的A、B、C或D
                            matches = re.findall(r'[ABCD]', response.upper())
                            if not matches:  # 检查是否找到任何匹配
                                raise Exception(f"No valid answer found in response: {response[:100]}")
                            answer_letter = matches[-1]

                        answer_index = ord(answer_letter) - ord('A')
                        if answer_index == correct_answer:
                            correct += 1
                        total += 1

                        response_info = {
                            "question": question,
                            "choices": choices,
                            "correct_answer": chr(correct_answer + ord('A')),
                            "model_answer": answer_letter,
                            "is_correct": answer_index == correct_answer,
                            "full_response": response
                        }
                        responses.append(response_info)

                        # 显示进度
                        accuracy = (correct / total) * 100
                        logging.info(f"Progress: {total}/{total_samples}, Current accuracy: {accuracy:.2f}%")

                    except Exception as e:
                        error_msg = f"Error processing question {idx}: {str(e)}"
                        logging.error(error_msg)
                        responses.append({
                            "question": question,
                            "error": error_msg,
                            "full_response": response if 'response' in locals() else None
                        })

                    # 每个问题后添加延迟
                    await asyncio.sleep(1)

            return {
                "subject": subject,
                "accuracy": (correct / total) * 100 if total > 0 else 0,
                "total_samples": total_samples,
                "completed_samples": total,
                "correct_answers": correct,
                "responses": responses
            }

        except Exception as e:
            error_msg = f"Error evaluating MMLU {subject}: {str(e)}"
            logging.error(error_msg)
            return {
                "subject": subject,
                "error": error_msg
            }

    async def evaluate_gsm8k(self) -> Dict[str, Any]:
        """评估GSM8K数学推理能力"""
        try:
            logging.info("Starting GSM8K evaluation")
            dataset = load_dataset("gsm8k", "main", split="test")
            total_samples = len(dataset)
            logging.info(f"Loaded {total_samples} test samples")

            correct = 0
            total = 0
            responses = []

            async with aiohttp.ClientSession() as session:
                for idx, item in enumerate(dataset):
                    question = item["question"]
                    correct_answer = item["answer"]

                    prompt = f"""请解决以下数学问题。请先逐步分析，然后在最后用"答案："标记您的最终答案（只需要写数字）：

{question}"""

                    try:
                        response = await self.get_model_response(session, prompt)
                        if response:
                            # 提取最后出现的数字
                            def extract_number(text):
                                numbers = re.findall(r'-?\d*\.?\d+', text)
                                return float(numbers[-1]) if numbers else None

                            # 首先尝试从"答案："后提取数字
                            final_answer_match = re.search(r'答案：\s*.*?(-?\d*\.?\d+)', response)
                            if final_answer_match:
                                model_number = float(final_answer_match.group(1))
                            else:
                                # 如果没有找到标记，从最后一行提取数字
                                model_number = extract_number(response.split('\n')[-1])

                            correct_number = extract_number(correct_answer)

                            if model_number is not None and correct_number is not None:
                                # 允许1%的相对误差
                                relative_error = abs(model_number - correct_number) / abs(correct_number)
                                is_correct = relative_error < 0.01

                                if is_correct:
                                    correct += 1
                                total += 1

                                response_info = {
                                    "question": question,
                                    "correct_answer": correct_number,
                                    "model_answer": model_number,
                                    "is_correct": is_correct,
                                    "full_response": response
                                }
                            else:
                                response_info = {
                                    "question": question,
                                    "error": "Could not extract valid number",
                                    "full_response": response
                                }

                            responses.append(response_info)

                            # 显示进度
                            accuracy = (correct / total) * 100 if total > 0 else 0
                            logging.info(f"Progress: {total}/{total_samples}, Current accuracy: {accuracy:.2f}%")

                            await asyncio.sleep(1)

                    except Exception as e:
                        logging.error(f"Error processing question {idx}: {str(e)}")
                        responses.append({
                            "question": question,
                            "error": str(e)
                        })
                        await asyncio.sleep(2)

            accuracy = (correct / total) * 100 if total > 0 else 0

            return {
                "accuracy": accuracy,
                "total_samples": total_samples,
                "completed_samples": total,
                "correct_answers": correct,
                "responses": responses
            }

        except Exception as e:
            logging.error(f"Error evaluating GSM8K: {str(e)}")
            return {"error": str(e)}

    async def evaluate_truthfulqa(self) -> Dict[str, Any]:
        """评估TruthfulQA数据集的表现"""
        try:
            logging.info("Starting TruthfulQA evaluation")
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
            total_samples = len(dataset)
            logging.info(f"Loaded {total_samples} test samples")

            correct = 0
            total = 0
            responses = []

            async with aiohttp.ClientSession() as session:
                for idx, item in enumerate(dataset):
                    question = item["question"]
                    choices = item["mc2_targets"]["choices"]
                    correct_answer = item["mc2_targets"]["labels"].index(1)

                    # 构建选项字符串
                    options_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    
                    prompt = f"""请回答以下问题，选择最准确的答案。请仔细分析每个选项，并在最后用"答案："标记您的选择（仅填写选项字母）：

问题：{question}

选项：
{options_str}"""

                    try:
                        response = await self.get_model_response(session, prompt)
                        if not response:
                            raise Exception("Received empty response from model")

                        # 提取答案
                        final_answer_match = re.search(r'答案：\s*([A-Z])', response.upper())
                        if final_answer_match:
                            answer_letter = final_answer_match.group(1)
                        else:
                            matches = re.findall(r'[A-Z]', response.upper())
                            if not matches:
                                raise Exception(f"No valid answer found in response: {response[:100]}")
                            answer_letter = matches[-1]

                        answer_index = ord(answer_letter) - ord('A')
                        if answer_index == correct_answer:
                            correct += 1
                        total += 1

                        response_info = {
                            "question": question,
                            "choices": choices,
                            "correct_answer": chr(correct_answer + ord('A')),
                            "model_answer": answer_letter,
                            "is_correct": answer_index == correct_answer,
                            "full_response": response
                        }
                        responses.append(response_info)

                        accuracy = (correct / total) * 100
                        logging.info(f"Progress: {total}/{total_samples}, Current accuracy: {accuracy:.2f}%")

                    except Exception as e:
                        error_msg = f"Error processing question {idx}: {str(e)}"
                        logging.error(error_msg)
                        responses.append({
                            "question": question,
                            "error": error_msg,
                            "full_response": response if 'response' in locals() else None
                        })

                    await asyncio.sleep(1)

            return {
                "task_name": "truthfulqa",
                "accuracy": (correct / total) * 100 if total > 0 else 0,
                "total_samples": total_samples,
                "completed_samples": total,
                "correct_answers": correct,
                "responses": responses
            }

        except Exception as e:
            error_msg = f"Error evaluating TruthfulQA: {str(e)}"
            logging.error(error_msg)
            return {
                "task_name": "truthfulqa",
                "error": error_msg
            }

    async def evaluate_bbh(self) -> Dict[str, Any]:
        """评估BBH(Big Bench Hard)数据集的表现"""
        try:
            logging.info("Starting BBH evaluation")
            
            # BBH任务列表
            bbh_tasks = [
                "boolean_expressions", "causal_judgement", "date_understanding",
                "formal_fallacies", "geometric_shapes", "hyperbaton",
                "logical_deduction_three_objects", "logical_deduction_five_objects",
                "logical_deduction_seven_objects", "movie_recommendation",
                "navigate", "object_counting", "penguins_in_a_table",
                "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
                "snarks", "sports_understanding", "temporal_sequences",
                "tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
                "tracking_shuffled_objects_seven_objects", "web_of_lies"
            ]

            all_results = []
            total_correct = 0
            total_questions = 0

            for task in bbh_tasks:
                try:
                    dataset = load_dataset("lukaemon/bbh", task, split="test")
                    task_correct = 0
                    task_total = 0
                    task_responses = []

                    async with aiohttp.ClientSession() as session:
                        for idx, item in enumerate(dataset):
                            input_text = item["input"]
                            target = item["target"]
                            
                            prompt = f"""请解决以下问题，并给出准确答案。在回答时，请直接给出答案，无需解释：

{input_text}"""

                            try:
                                response = await self.get_model_response(session, prompt)
                                if not response:
                                    raise Exception("Received empty response from model")

                                # 清理响应文本
                                cleaned_response = response.strip().lower()
                                cleaned_target = target.strip().lower()

                                is_correct = cleaned_response == cleaned_target
                                if is_correct:
                                    task_correct += 1
                                task_total += 1

                                response_info = {
                                    "input": input_text,
                                    "target": target,
                                    "model_response": response,
                                    "is_correct": is_correct
                                }
                                task_responses.append(response_info)

                            except Exception as e:
                                error_msg = f"Error processing question {idx}: {str(e)}"
                                logging.error(error_msg)
                                task_responses.append({
                                    "input": input_text,
                                    "error": error_msg
                                })

                            await asyncio.sleep(1)

                    task_accuracy = (task_correct / task_total) * 100 if task_total > 0 else 0
                    task_result = {
                        "task_name": task,
                        "accuracy": task_accuracy,
                        "total_samples": len(dataset),
                        "completed_samples": task_total,
                        "correct_answers": task_correct,
                        "responses": task_responses
                    }
                    all_results.append(task_result)
                    total_correct += task_correct
                    total_questions += task_total

                except Exception as e:
                    error_msg = f"Error evaluating BBH task {task}: {str(e)}"
                    logging.error(error_msg)
                    all_results.append({
                        "task_name": task,
                        "error": error_msg
                    })

            return {
                "task_name": "bbh",
                "average_accuracy": (total_correct / total_questions) * 100 if total_questions > 0 else 0,
                "total_tasks": len(bbh_tasks),
                "total_questions": total_questions,
                "total_correct": total_correct,
                "task_results": all_results
            }

        except Exception as e:
            error_msg = f"Error evaluating BBH: {str(e)}"
            logging.error(error_msg)
            return {
                "task_name": "bbh",
                "error": error_msg
            }

    async def evaluate_dataset(self, task_name: str, dataset) -> Dict[str, Any]:
        """评估数据集并实时更新结果"""
        try:
            start_time = time.time()
            logging.info(f"Starting evaluation of {task_name} with {len(dataset)} samples")
            correct = 0
            total = 0
            
            async with await self.create_session() as session:
                for idx, item in enumerate(dataset):
                    question_start_time = time.time()
                    try:
                        # 处理问题数据
                        if "question" in item and "choices" in item and "answer" in item:
                            question_type = "multiple_choice"
                            question = item["question"]
                            choices = item["choices"]
                            correct_answer = item["answer"]
                            
                            prompt = f"""Question: {question}\n\nOptions:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnalyze each option and provide your answer with "Answer:" followed by only A, B, C, or D."""
                            
                        elif "question" in item and "answer" in item:
                            question_type = "free_form"
                            question = item["question"]
                            correct_answer = item["answer"]
                            
                            prompt = f"""Question: {question}\n\nProvide your answer after "Answer:"."""
                            
                        else:
                            raise ValueError(f"Unsupported dataset format for task {task_name}")

                        # 获取模型响应
                        response = await self.get_model_response(session, prompt)
                        
                        # 处理响应
                        if question_type == "multiple_choice":
                            answer_match = re.search(r'Answer:\s*([ABCD])', response.upper())
                            model_answer = answer_match.group(1) if answer_match else re.findall(r'[ABCD]', response.upper())[-1]
                            answer_index = ord(model_answer) - ord('A')
                            is_correct = (answer_index == correct_answer)
                            
                            question_result = {
                                "index": idx,
                                "question": question,
                                "options": choices,
                                "correct_answer": chr(correct_answer + ord('A')),
                                "model_answer": model_answer,
                                "is_correct": is_correct
                            }
                        else:
                            model_answer = response.strip()
                            if isinstance(correct_answer, (int, float)) or re.match(r'^-?\d*\.?\d+$', str(correct_answer)):
                                try:
                                    model_number = float(re.findall(r'-?\d*\.?\d+', model_answer)[-1])
                                    correct_number = float(str(correct_answer))
                                    is_correct = abs(model_number - correct_number) / abs(correct_number) < 0.01
                                except:
                                    is_correct = False
                            else:
                                is_correct = model_answer.lower() == str(correct_answer).lower()
                            
                            question_result = {
                                "index": idx,
                                "question": question,
                                "correct_answer": str(correct_answer),
                                "model_answer": model_answer,
                                "is_correct": is_correct
                            }

                        if is_correct:
                            correct += 1
                        total += 1

                        # 计算处理时间
                        processing_time = time.time() - question_start_time
                        question_result.update({
                            "processing_time": processing_time,
                            "questions_per_second": 1 / processing_time
                        })

                        # 更新结果
                        self.update_results(task_name, question_result)

                        # 记录详细进度日志
                        accuracy = (correct / total) * 100
                        speed = 1 / processing_time
                        logging.info(
                            f"Task: {task_name} | "
                            f"Progress: {total}/{len(dataset)} ({total/len(dataset)*100:.1f}%) | "
                            f"Accuracy: {accuracy:.2f}% | "
                            f"Speed: {speed:.2f} questions/s | "
                            f"Question {idx+1}: {'✓' if is_correct else '✗'}"
                        )

                    except Exception as e:
                        error_msg = f"Error in {task_name} sample {idx}: {str(e)}"
                        logging.error(error_msg)
                        question_result = {
                            "index": idx,
                            "question": question if 'question' in locals() else "Unknown",
                            "error": str(e),
                            "processing_time": time.time() - question_start_time
                        }
                        self.update_results(task_name, question_result)

                    await asyncio.sleep(0.5)

            # 更新任务完成状态
            total_time = time.time() - start_time
            final_result = {
                "task_name": task_name,
                "total_questions": len(dataset),
                "completed_questions": total,
                "correct_answers": correct,
                "accuracy": (correct / total * 100) if total > 0 else 0,
                "total_time": total_time,
                "average_speed": total / total_time if total_time > 0 else 0
            }
            
            self.update_results(task_name, final_result, is_task_complete=True)
            
            logging.info(
                f"Completed {task_name} | "
                f"Final Accuracy: {final_result['accuracy']:.2f}% | "
                f"Total Time: {total_time:.2f}s | "
                f"Average Speed: {final_result['average_speed']:.2f} questions/s"
            )

            return final_result

        except Exception as e:
            error_msg = f"Task {task_name} evaluation failed: {str(e)}"
            logging.error(error_msg)
            return {
                "task_name": task_name,
                "error": str(e)
            }

    def initialize_results_file(self):
        """Initialize the results JSON file with basic structure"""
        try:
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)

            # 使用固定的文件名
            self.output_file = output_dir / "evaluation_results.json"
            
            initial_structure = {
                "evaluation_info": {
                    "model_name": self.model_name,
                    "start_time": int(time.time()),
                    "api_url": self.api_url
                },
                "progress": {
                    "total_tasks": 0,
                    "completed_tasks": 0,
                    "current_task": "",
                    "status": "running"
                },
                "results": {
                    "tasks": {},
                    "questions": {}
                }
            }

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(initial_structure, f, ensure_ascii=False, indent=2)

            logging.info(f"Initialized results file: {self.output_file}")

        except Exception as e:
            logging.error(f"Error initializing results file: {str(e)}")
            raise

    def update_results(self, task_name: str, question_data: Dict[str, Any], is_task_complete: bool = False):
        """实时更新结果文件"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # 更新问题记录
            if task_name not in results["results"]["questions"]:
                results["results"]["questions"][task_name] = []
            
            # 添加时间戳和速度信息
            current_time = int(time.time())
            if results["results"]["questions"][task_name]:
                last_question = results["results"]["questions"][task_name][-1]
                time_diff = current_time - last_question["timestamp"]
                question_data["processing_time"] = time_diff
            
            question_data["timestamp"] = current_time
            results["results"]["questions"][task_name].append(question_data)

            # 更新任务进度
            results["progress"]["current_task"] = task_name
            
            # 如果任务完成，更新任务结果
            if is_task_complete:
                if task_name not in results["results"]["tasks"]:
                    results["results"]["tasks"][task_name] = {}
                
                task_questions = results["results"]["questions"][task_name]
                correct_count = sum(1 for q in task_questions if q.get("is_correct", False))
                total_count = len(task_questions)
                
                results["results"]["tasks"][task_name].update({
                    "total_questions": total_count,
                    "correct_answers": correct_count,
                    "accuracy": (correct_count / total_count * 100) if total_count > 0 else 0,
                    "completion_time": current_time
                })

                results["progress"]["completed_tasks"] += 1

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.error(f"Error updating results: {str(e)}")

    def setup_logging(self):
        """配置详细的日志记录"""
        try:
            log_dir = Path("evaluation_logs")
            log_dir.mkdir(exist_ok=True)

            # 使用日期作为日志文件名
            log_file = log_dir / f"evaluation_{time.strftime('%Y%m%d')}.log"

            # 配置日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # 清除现有的处理器
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # 文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # 配置根日志记录器
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            logging.info(f"Starting evaluation with model: {self.model_name}")
            logging.info(f"API URL: {self.api_url}")

        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            raise

    async def run_evaluation(self) -> None:
        """Run all evaluations with continuous result writing"""
        try:
            # Initialize results file
            self.initialize_results_file()

            # MMLU subjects list (unchanged)
            mmlu_subjects = [
                # Mathematics and Logic
                "abstract_algebra", "college_mathematics", "elementary_mathematics",
                "formal_logic", "high_school_mathematics", "mathematical_logic",

                # Science
                "anatomy", "astronomy", "biology", "chemistry", "college_biology",
                "college_chemistry", "college_physics", "computer_science",
                "conceptual_physics", "electrical_engineering", "high_school_biology",
                "high_school_chemistry", "high_school_physics", "machine_learning",

                # Social Sciences
                "econometrics", "high_school_geography", "high_school_government_and_politics",
                "high_school_macroeconomics", "high_school_microeconomics",
                "high_school_psychology", "human_sexuality", "professional_psychology",
                "public_relations", "sociology", "us_foreign_policy",

                # Humanities
                "high_school_european_history", "high_school_us_history",
                "high_school_world_history", "international_law", "jurisprudence",
                "philosophy", "world_religions",

                # Professional
                "business_ethics", "clinical_knowledge", "college_medicine",
                "human_aging", "management", "marketing", "medical_genetics",
                "professional_accounting", "professional_law", "professional_medicine",

                # Miscellaneous
                "moral_disputes", "moral_scenarios", "nutrition", "prehistory",
                "security_studies", "virology"
            ]

            # 定义所有评估任务
            evaluation_tasks = {
                "mmlu": mmlu_subjects,
                "gsm8k": None,
                "truthfulqa": None,
                "bbh": None
            }

            # Update total tasks in results file
            total_tasks = len(mmlu_subjects) + 3  # MMLU subjects + GSM8K + TruthfulQA + BBH
            with open(self.output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)
            current_results["progress"]["total_tasks"] = total_tasks
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)

            # First evaluate MMLU subjects in parallel batches
            for i in range(0, len(mmlu_subjects), 4):
                batch_subjects = mmlu_subjects[i:i + 4]
                tasks = []

                for subject in batch_subjects:
                    try:
                        dataset = load_dataset("cais/mmlu", subject, split="test")
                        tasks.append(self.evaluate_dataset(f"mmlu_{subject}", dataset))
                    except Exception as e:
                        logging.error(f"Error loading dataset for {subject}: {str(e)}")
                        self.update_results(f"mmlu_{subject}", {
                            "task_name": f"mmlu_{subject}",
                            "error_message": str(e)
                        })
                        continue

                if tasks:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for subject, result in zip(batch_subjects, batch_results):
                        if isinstance(result, Exception):
                            logging.error(f"Error evaluating {subject}: {str(result)}")
                            self.update_results(f"mmlu_{subject}", {
                                "task_name": f"mmlu_{subject}",
                                "error_message": str(result)
                            })
                        else:
                            self.update_results(f"mmlu_{subject}", result)
                            self._update_progress_and_average()

                await asyncio.sleep(1)

            # Then evaluate other tasks sequentially
            # GSM8K
            try:
                gsm8k_result = await self.evaluate_gsm8k()
                self.update_results("gsm8k", gsm8k_result)
                self._update_progress_and_average()
            except Exception as e:
                logging.error(f"Error evaluating GSM8K: {str(e)}")
                self.update_results("gsm8k", {"error": str(e)})

            # TruthfulQA
            try:
                truthfulqa_result = await self.evaluate_truthfulqa()
                self.update_results("truthfulqa", truthfulqa_result)
                self._update_progress_and_average()
            except Exception as e:
                logging.error(f"Error evaluating TruthfulQA: {str(e)}")
                self.update_results("truthfulqa", {"error": str(e)})

            # BBH
            try:
                bbh_result = await self.evaluate_bbh()
                self.update_results("bbh", bbh_result)
                self._update_progress_and_average()
            except Exception as e:
                logging.error(f"Error evaluating BBH: {str(e)}")
                self.update_results("bbh", {"error": str(e)})

            # Mark evaluation as complete
            with open(self.output_file, 'r', encoding='utf-8') as f:
                final_results = json.load(f)

            final_results["progress"]["status"] = "completed"
            final_results["progress"]["completion_time"] = int(time.time())

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            logging.info("Evaluation completed")

        except Exception as e:
            logging.error(f"Error in evaluation suite: {str(e)}")
            self.update_results("error", {
                "error_message": str(e),
                "timestamp": int(time.time())
            })

    def _update_progress_and_average(self):
        """更新进度和平均分数"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                current_results = json.load(f)

            # 计算已完成的任务数
            completed_tasks = 0
            if "mmlu" in current_results["results"]["tasks"]:
                completed_tasks += len(current_results["results"]["tasks"]["mmlu"])
            for task in ["gsm8k", "truthfulqa", "bbh"]:
                if task in current_results["results"]["tasks"] and "error" not in current_results["results"]["tasks"][task]:
                    completed_tasks += 1

            # 更新进度信息
            current_results["progress"]["completed_tasks"] = completed_tasks
            current_results["progress"]["status"] = "running"
            current_results["progress"]["last_update"] = int(time.time())

            # 计算总体平均分数
            total_accuracy = 0
            total_tasks = 0

            # MMLU平均分数
            if "mmlu" in current_results["results"]["tasks"]:
                valid_results = [r["accuracy"]
                               for r in current_results["results"]["tasks"]["mmlu"]
                               if "accuracy" in r]
                if valid_results:
                    total_accuracy += sum(valid_results)
                    total_tasks += len(valid_results)
                    current_results["results"]["tasks"]["mmlu"]["average_accuracy"] = sum(valid_results) / len(valid_results)

            # 其他任务的分数
            for task in ["gsm8k", "truthfulqa", "bbh"]:
                if task in current_results["results"]["tasks"] and "accuracy" in current_results["results"]["tasks"][task]:
                    total_accuracy += current_results["results"]["tasks"][task]["accuracy"]
                    total_tasks += 1

            # 更新总体平均分数
            if total_tasks > 0:
                current_results["results"]["tasks"]["average_accuracy"] = total_accuracy / total_tasks

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(current_results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logging.error(f"Error updating progress and average: {str(e)}")

if __name__ == "__main__":
    try:
        evaluator = RemoteModelEvaluator()
        asyncio.run(evaluator.run_evaluation())
        logging.info("Evaluation completed!")
    except Exception as e:
        logging.error(f"Runtime error: {str(e)}")
