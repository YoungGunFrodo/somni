# Using your sleep-learning model

This guide is for running the real system on your own PC: talk to the model, teach it, let it
sleep, and approve what it learned. Tested on Windows 11 + AMD RX 6800 (16 GB).

## 1. One-time setup

```
cd sleep-learning-ai
py -3.13 -m venv .venv-gpu
.venv-gpu\Scripts\pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1030]"
.venv-gpu\Scripts\pip install numpy transformers peft accelerate huggingface_hub safetensors
```

(NVIDIA: install the CUDA build of torch instead. CPU-only works, but training is slow.)

Create your model's home folder and config. Everything it learns lives in this folder
(default `%USERPROFILE%\.sleep-ai`), never in the repo:

```
.venv-gpu\Scripts\python -m sleepai.cli status          # creates the folder and config.json
```

Edit `%USERPROFILE%\.sleep-ai\config.json` and set at least:

```json
{
  "backend": "real",
  "trainer": "peft",
  "base_model": "Qwen/Qwen3-1.7B",
  "wake": "hf",
  "teacher_url": "",
  "teacher_model": ""
}
```

- `base_model`: any Hugging Face chat model that fits your GPU for LoRA training in bf16:
  up to about 4B parameters on 16 GB. For coding, try `Qwen/Qwen2.5-Coder-3B-Instruct` or
  `Qwen/Qwen3-4B`. The model downloads on first use. Changing it later means a brain transfer.
- `wake: "hf"` runs the model inside Python (no server). `"server"` uses llama-server with the
  approved adapter converted to GGUF (faster replies; see section 6).
- `teacher_url` empty: the model teaches itself during sleep. To use a bigger teacher from LM
  Studio, start LM Studio's server and set `"teacher_url": "http://127.0.0.1:1234"` and
  `"teacher_model"` to the model id shown by `lms ls`. Stop other GPU work while it sleeps.

Check everything: `.venv-gpu\Scripts\python -m sleepai.cli doctor`

## 2. Wake: talk to it and teach it

```
.venv-gpu\Scripts\python -m sleepai.cli chat
```

- Ask anything ending with `?`. It searches its memory first and answers.
- State facts, preferences, decisions as plain sentences: `The deploy script for my blog is
  deploy.ps1.` / `I prefer pytest over unittest.` They are logged as episodes.
- If an answer is wrong: `/wrong The port of my dev server is 5173.` (a correction, highest
  priority). If right: `/right`.
- `tell "<text>"` does the same from the command line without starting a chat.

New memories are formed during sleep. Until then the model has not "learned" them.

## 3. Sleep

Run one sleep now:

```
.venv-gpu\Scripts\python -m sleepai.cli sleep
```

or let it sleep by itself when the PC is idle (30 minutes) or between 1:00 and 7:00:

```
.venv-gpu\Scripts\python -m sleepai.cli daemon
```

A sleep forms memories from the day's episodes, trains a LoRA adapter (with replay of older
memories), dreams (checked practice on what it got wrong), runs every gate, and writes a dream
report. Long nights are fine: more practice examples per memory means deeper learning. To wake it
early, create a file named `WAKE` in the home folder; it stops at the next safe point.

## 4. Morning review

```
.venv-gpu\Scripts\python -m sleepai.cli review
```

shows what waits and the path of the dream report (`sleeps\sleep-NNNN\report.md`). Read it: new
memories, conflicts, and the gate table. Then:

```
...cli review approve                         # adopt the new checkpoint
...cli review reject                          # keep the current one
...cli review reject-memories <id> <id> ...   # a memory is wrong: never train it
...cli review resolve <id> accept|reject      # decide a disputed memory
...cli review rollback                        # undo the last approved checkpoint
```

Nothing changes the model's weights without your approval (autonomy level 0).

## 5. Check progress

`...cli status` shows the active checkpoint, memories (protected / condensed / faded), and the
last dream report. The gates in each report show learning on new memories, retention of old
ones, general knowledge, coding tasks (Python tests actually run), and calibration.

## 6. Faster replies with llama-server (optional)

Convert the approved adapter and base model to GGUF and serve them with the llama.cpp Vulkan
build in `tools/` (see `scripts/milestone0_llamacpp.py` for the exact commands), then set
`"wake": "server"` and `"llamacpp_url": "http://127.0.0.1:8080"`.

## Coding

What it can learn from your coding work today: facts and preferences about your projects
(paths, commands, conventions, which library you use for what), decisions and their reasons,
and corrections. The coding gate makes sure learning these never breaks its general coding
ability. Not built yet: learning new coding SKILLS from your sessions (dreaming on code tasks
with unit-test checkers); the checker exists, the curriculum does not.

## Where things are

```
%USERPROFILE%\.sleep-ai\
  config.json       settings
  memory.db         episodes and the memory store (the durable asset; back it up)
  checkpoints\      ckpt-0000 (base), ckpt-0001, ... each with its adapter
  sleeps\           one folder per sleep with report.md
```


## Skills the model keeps (no setup needed)

While it learns your memories, the model rehearses its own earlier answers so it keeps what it
could already do. Four skill packs are protected for everyone: general knowledge, reasoning,
writing and coding. How much of each is rehearsed follows what you actually do: each sleep looks
at your last 30 days of conversations, and half the rehearsal is split by that usage, half evenly
(so a skill you have not used lately is still kept). The dream report shows the mix.

To add your own domain (medicine, law, a language, a game), register a pack in your own code:

```python
from sleepai.skills import SkillPack, register
register(SkillPack("chess", rehearsal=["What is the Sicilian Defence?", "..."],
                   detect=lambda text: "chess" in text.lower()))
```
and add `"chess"` to `"skills"` in `config.json`.
