Some LLM inference demos using accelerate.



* acc_demo_1.py

  A easy demo.

* acc_demo_llm.py

  一个简单的demo从网上copy的

* acc_demo_llm_batch.py

  在上面基础上实现一个运用批处理的demo，但貌似是多个gpu上都部署了模型，然后将批处理按GPU分给每个GPU，在每一个机器上单独跑，但性能很好，GPU利用率基本保证100%。

* acc_demo_llm_batch_pipeline.py

  一个手动的修改的批处理的流水线demo。通过将device_map设置为auto，在每一层分割模型，实现了流水线并行。

  只在GPU0上进行输入输出（貌似这里可以在多个GPU上都进行输入输出，这样可能更能发挥各机器的性能，但不多）。

  GPU利用率在四成到七成。

* llama.py

  accelerate's demo using pippy to achieve pipeline parallelization.