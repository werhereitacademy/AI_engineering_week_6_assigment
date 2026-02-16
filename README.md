**Assignment: Fine-Tune a Small LLM for Better Function Calling – Agentic Assistant Edition**

**Goal:** Fine-tune a small open-source LLM to improve its ability to decide when to call tools (functions) in JSON format vs. responding directly. This makes it more "agentic" – e.g., using a calculator tool for math instead of guessing. You'll measure success with a simple accuracy score and experiment with hyperparameters.

**Why this?**  
Function calling is key for agents (like in LangChain). Base models often hallucinate or format JSON wrong. Fine-tuning teaches patterns from examples. You'll use Unsloth + TRL for speed, and evaluate to prove improvement (aim for 20–40% accuracy gain).

**Requirements**  
- **Colab:** Free tier (T4 GPU).  
- **Model:** Pick **one small model** (all ~2–4B params, train in 20–45 min):  
  - `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (recommended – strong for tools, low VRAM)  
  - `unsloth/gemma-2-2b-it-bnb-4bit` (Gemma-2 variant, fast & capable)  
- **Dataset:** `glaiveai/glaive-function-calling-v2` (open-source, ~20k tool-calling examples).  
  - Subsample: Train on 2,000–2,500 shuffled examples (`dataset.shuffle(seed=42).select(range(2500))`).  
  - Eval: Hold out next 200 (`select(range(2500, 2700))`) for testing.  
- **Max Seq Length:** Start with 2048 (adjust if OOM; try 1024 for speed).  
- **Libraries:** Unsloth + TRL SFTTrainer (install as in class resources).

**Step-by-Step Guide**  
1. **Setup & Load Model** (Copy from Unsloth docs; adapt model name)  
   ```python
   !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   from unsloth import FastLanguageModel
   import torch
   from datasets import load_dataset
   from trl import SFTTrainer
   from transformers import TrainingArguments

   max_seq_length = 2048  # Start here; tune if needed (e.g., 1024 for faster)
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # Or gemma-2-2b
       max_seq_length=max_seq_length,
       dtype=None,
       load_in_4bit=True,
   )
   model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=42)
   ```

2. **Load & Prepare Dataset** (Key: Chat Template – Be Careful!)
   

3. **Train with Defaults (Tune to Improve)**  
   Start with these **default params** (good baseline; ~20–40 min on T4):  
   ```python
   trainer = SFTTrainer(
       model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_ds,
       dataset_text_field="text", max_seq_length=max_seq_length,
       dataset_num_proc=2, packing=True,  # Speed boosts
       args=TrainingArguments(
           per_device_train_batch_size=2,
           gradient_accumulation_steps=4,
           warmup_steps=10,
           max_steps=120,  # Default; tune to 200–300 for better results
           learning_rate=2e-4,  # Default; try 1e-4 or 5e-4
           fp16=not torch.cuda.is_bf16_supported(),
           bf16=torch.cuda.is_bf16_supported(),
           logging_steps=10,
           optim="adamw_8bit",
           weight_decay=0.01,
           lr_scheduler_type="linear",
           evaluation_strategy="steps", eval_steps=50,
           seed=42, output_dir="outputs",
       ),
   )
   trainer.train()
   trainer.save_model("my_function_calling_model")
   ```
   **Tuning Challenge:** Run once with defaults. Then experiment:  
   - Increase `max_steps=200` (more training, better accuracy?).  
   - Try `learning_rate=1e-4` (slower but stable?).  
   - Reduce `max_seq_length=1024` if slow/OOM.  
   Report what worked best!

4. **Evaluate: Prove Improvement**  
   Measure success rate of right tool calling: % correct tool decisions/formats.  

   # Run before/after

   base_acc = eval_accuracy(model, tokenizer, eval_ds)  # Base model

   # Fine-tune, then: tuned_acc = eval_accuracy(model, tokenizer, eval_ds)
   print(f"Base: {base_acc:.1f}% | Tuned: {tuned_acc:.1f}%")  # Expect tuned > base by 20%+
   ```
   - Test 3–5 manual prompts too (e.g., "Calculate 10*7" → JSON add/multiply; "What's fun about cats?" → Text response).  
   - Success: Tuned accuracy 60–80% (base ~30–50%).

**Deliverables (PDF/Doc + Public Colab Link)**  
- Colab notebook (end-to-end, with your tuned params).  
- Screenshots: Loss curve, accuracy scores (base vs. tuned), 2 example gens (JSON correct?).  
- Report:  
  1. Model/dataset choice & subsample size?  
  2. Default vs. tuned params (what changed? Better accuracy?).  
  3. Improvement evidence (accuracy % + 1–2 examples: e.g., base hallucinated "73" → tuned JSON call).  
  4. Challenges (e.g., template bugs, tuning fails)?
