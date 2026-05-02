# Cathey Final Presentation — Speaker Scripts

EECS 6895 Final Project · Total time: ~12 minutes (10 min talk + ~2 min demo/QA buffer)

Three speakers. Roughly equal share of slides.

- **Speaker 1** (Slides 1–5): Motivation, novelty, system overview, tech stack. ~3 min 30 sec.
- **Speaker 2** (Slides 6–10): Data and methods (unified JSON, hybrid pipeline, memory, LoRA). ~3 min 30 sec.
- **Speaker 3** (Slides 11–15): Experiments, demo, and conclusion. ~3 min 30 sec.

Each speaker's words are written in simple English that is easy to remember and easy to pronounce on stage. Bracketed `[...]` notes are stage cues, not lines to read out.

---

## Speaker 1 — Slides 1 to 5

### [Slide 1 — Title]

Hello everyone. We are group [number]. Today we are presenting our EECS 6895 final project, called Nova. Cathey is an offline voice-controlled smart home assistant that runs on a Raspberry Pi 5. I am [Name 1], and my teammates [Name 2] and [Name 3] will continue after me.

### [Slide 2 — Motivation]

So, why did we build another voice assistant? Today, almost every voice assistant on the market — Alexa, Google Assistant, Siri — depends on the cloud. Every word you say is sent to a remote server, and you cannot really control where your data goes. If your internet is down, the assistant simply stops working.

We wanted to ask a different question: can a small, eighty-dollar device like the Raspberry Pi 5 do all of this work on its own? No cloud, no internet, no streaming.

So our goal was to build Nova: a voice assistant where everything — speech-to-text, intent parsing, dialogue, text-to-speech, and even the GPIO control of real hardware — happens on the Pi itself.

### [Slide 3 — Novelty]

What makes Nova different from a typical class project?

First, we use a hybrid pipeline. A simple regex catches around seventy percent of the common commands in less than five milliseconds. The LLM is only called when the user says something vague or unusual.

Second, we ran a controlled study of five GGUF quantization variants of Qwen2.5 on the same Pi. Most papers test quantization on standard benchmarks; we test it on a real edge task.

Third, we designed a four-layer memory: working, episodic, semantic, and procedural. The system actually learns what the user prefers and skips repeated clarification questions.

Fourth, we did LoRA fine-tuning on a hand-built dataset of 225 examples for four intent classes.

And finally, we have a real hardware demo — an LED ring and a stepper-motor curtain, both controlled directly through the Pi's GPIO pins.

### [Slide 4 — System Pipeline]

Here is the full pipeline. Audio comes in from the microphone and goes through a voice activity detector. Whisper, the speech-to-text model, turns it into text. Then a wake-word filter checks for the word "Cathey" or one of its variants.

After that, the text first goes to the rule-based fast path. If the regex matches, we return the command immediately. If not, we fall back to the LLM. The agent dispatches the result based on its intent type, updates memory, validates the command, and finally runs the GPIO action and the TTS reply in parallel threads, so the user hears the reply at the same time as the device acts.

Everything you see in this diagram runs on a single Raspberry Pi 5, fully offline.

### [Slide 5 — Tech Stack]

A quick look at the stack. On the hardware side: Pi 5 with eight gigabytes of RAM, a ReSpeaker microphone HAT, a Grove RGB LED, a stepper motor for the curtain, a USB mic, and a Bluetooth speaker.

For models: faster-whisper "tiny.en" in int8 for STT, Qwen2.5-3B-Instruct in Q3_K_M GGUF for the LLM, all-MiniLM-L6-v2 for embeddings, and Piper for TTS.

For software: llama-cpp-python with OpenBLAS on ARM, HuggingFace transformers and PEFT for fine-tuning, ChromaDB for the vector store, and systemd to auto-start the assistant on boot.

I will now hand over to [Name 2], who will go through the data and the methods.

---

## Speaker 2 — Slides 6 to 10

### [Slide 6 — Data]

Thank you, [Name 1]. So, the first problem we faced was that there is no public dataset that exactly matches our task. We needed labelled examples for four intent classes in English, with our specific four devices: light, curtain, window, and AC. Public datasets like SLURP cover device control, but they don't have the "feeling" or the "general question" classes that we care about.

So we built our own. Two hundred and twenty-five hand-labelled pairs. Each pair is one user utterance and one JSON output. The split is shown here: seventy-six direct commands, sixty-five clarifications, sixty general QA, and twenty-four invalid examples.

We made sure to include hard cases. Things like "fuck this light" — colloquial complaints. Things like "I feel cold" — vague feelings that should never be turned into a direct command. And confusing cases like "can I still eat this dish from yesterday?", which mentions food and the fridge but is a general question, not a device command.

Each example was reviewed by two team members.

### [Slide 7 — Unified JSON]

Now, the methods. The first design choice we made was that all four intent types share one output schema. The LLM always returns one JSON object, with a "type" field that tells the agent what to do.

You can see the four shapes here: direct command, needs clarification, general QA, and invalid. The agent code uses one dispatch table on the type field, instead of branching on different formats. This also makes LoRA fine-tuning simpler, because the loss is computed on a single, well-defined output.

### [Slide 8 — Hybrid Rule + LLM]

Our second design choice was the hybrid pipeline. We made one observation: in real home control, the long tail is small. People say things like "turn on the light" or "set AC to 24" most of the time. We can cover all of that with eight or ten regex patterns.

So before we touch the LLM, we run the regex. If it matches, we return the command in less than five milliseconds. If it doesn't match, we fall back to the LLM, which on the Pi takes about four seconds.

The bottom of the slide shows the effect on the five direct-command cases of our benchmark: from an average of 4.2 seconds to 4 milliseconds. That's roughly a thousand-times speedup on the most common request type.

### [Slide 9 — Four-Layer Memory]

Method three is the four-layer memory. Working memory is just a deque of the last eight turns, in RAM. Episodic memory is a ChromaDB vector store of past interactions, used for retrieval. Semantic memory is a JSON file of user preferences, like the preferred AC temperature. And procedural memory is a JSON file of trigger-action pairs that the system has learned.

Procedural memory is the most interesting one. Imagine on day one, you say "I feel cold". Nova asks you whether to close the window or raise the AC. You answer "close the window". Nova saves that pair.

Day two, you say "I feel cold" again. This time Nova doesn't ask. It just closes the window directly, because it remembers your past choice. The lookup is done with cosine similarity on sentence embeddings, with a threshold of 0.92.

### [Slide 10 — LoRA]

Method four is LoRA fine-tuning. We use rank eight, alpha sixteen, on seven attention and MLP modules of Qwen2.5. About 0.44 percent of the parameters are trainable, which is roughly 6.8 million out of 1.5 billion.

We train for three epochs with a small batch size and gradient accumulation, using TRL's SFTTrainer with response-only loss masking. The final adapter is around fifty megabytes. We then merge it into the base weights and export to GGUF, so we can run it on the Pi with llama.cpp.

I will now pass over to [Name 3], who will show the experiments and the demo.

---

## Speaker 3 — Slides 11 to 15

### [Slide 11 — Quantization Benchmark]

Thanks, [Name 2]. Now let me show you the experiments.

The first one is the quantization benchmark. We compared five GGUF variants of Qwen2.5: three sizes of the 1.5B model and two sizes of the 3B model. All on the same Pi 5, same context length, same temperature, three repetitions per case to get the median latency.

The result is here. The 3B Q3_K_M reaches 85 percent type accuracy, with about 3.9 seconds average latency, and only 1.5 gigabytes on disk. It clearly beats every 1.5B variant. It also beats the 3B Q4_K_M, which is bigger and slower.

The main lesson is that, for small intent tasks, model size matters more than quantization level. So we picked the 3B Q3_K_M as our production model.

### [Slide 12 — LoRA Effect]

The second experiment shows the effect of LoRA. We took six hard cases that the base 1.5B model gets wrong. After LoRA fine-tuning on our 225 examples, all six cases are corrected.

The most interesting fixes are the colloquial ones — "fuck this light" or "make this room lively". The base model has never seen enough examples to handle them. LoRA learned them from a few dozen hand-written examples.

### [Slide 13 — Latency Budget]

The third experiment is the end-to-end latency on the Pi.

For a direct command, the total is about 2.3 seconds: half a second for the VAD, 0.6 seconds for STT, less than five milliseconds for the regex, and 1.2 seconds for GPIO and TTS in parallel.

For a vague utterance like "I feel cold", the LLM call replaces the regex and takes around four seconds, so the total is closer to six seconds.

But once procedural memory kicks in — the second time the user says "I feel cold" — we skip the LLM entirely and we are back to 2.3 seconds.

### [Slide 14 — Demo]

[Optional: switch to live demo or video.]

I would like to show a short live demo of the system. We have the Pi here with the LED ring and the stepper-motor curtain.

[Trigger demo. Speak each line clearly toward the Pi, wait for a moment so the audience can see the device respond.]

- "Nova, turn on the light." — The LED ring lights up.
- "Nova, set brightness to 30 percent." — The light dims.
- "Nova, start RGB mode." — The LED cycles through colors.
- "Nova, close the curtain." — The stepper motor moves.
- "Nova, I feel cold." — Nova asks a clarification.
- I reply: "close the window." — Nova closes it and saves the pattern.
- "Nova, how do I store leftovers?" — Nova answers in plain speech.

Everything you just saw is offline.

### [Slide 15 — Conclusion]

To summarize. We built a fully offline voice assistant on a Raspberry Pi 5. It reaches 85 percent type accuracy with 3.9 seconds average LLM latency. The rule-based path gives a roughly 1000-times speedup for direct commands. LoRA fixes all the hard cases. The procedural memory removes repeated clarifications.

The three main lessons are: model size matters more than quantization level for small intent tasks; hybrid pipelines are the right answer on edge devices; and a small hand-labelled dataset is enough for LoRA on a 3B chat model, as long as it covers all four intent classes.

For future work we would like to add streaming TTS, so the user hears the reply as it is being generated; time-aware procedural memory, so the system knows whether it is morning or night; and a distilled model under half a billion parameters so the LLM call drops below one second.

Thank you for listening. We are happy to take any questions.

---

## Q&A Buffer (likely questions and short answers)

- **Why Qwen2.5 and not LLaMA-3?** LLaMA-3's smallest version is 8B, which does not fit comfortably with our latency budget on Pi 5. Qwen2.5 has a clean 3B GGUF release and produces strict JSON reliably.
- **Why a regex layer instead of just trusting the LLM?** Latency. The LLM is 4 seconds; the regex is 5 milliseconds. For the 70% of commands that are unambiguous, the LLM adds nothing.
- **How do you handle privacy on the Pi itself?** Everything is local. The audio buffer is processed in RAM (no temp WAV files written to disk). Episodic memory is local to the device. Nothing leaves the Pi.
- **Why only 225 training samples?** Because LoRA on a 3B model with response-only loss masking does not need much data. Adding more samples is a clear next step.
- **What if the user has an accent and Whisper mishears "Cathey"?** We added a list of variants ("nava", "noba", "noa", ...). A dedicated wake-word detector like openWakeWord would be the long-term fix.
- **Does the LLM call run on GPU?** No. The Pi 5 has no usable GPU for LLMs. Everything is on the 4-core Cortex-A76 CPU through llama.cpp with OpenBLAS.
