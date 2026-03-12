Step 1 — Collect images — You gather photos or scans of handwritten Odia text. This is the most important step. More images = better model.
Step 2 — Label the data — For every image you tell the model "this image contains this Odia text." For example, image of "ନମସ୍କାର" → you label it as ନମସ୍କାର. This is called ground truth.
Step 3 — Preprocess — TrOCR needs images in a specific format. The TrOCR processor automatically handles resizing and converting for you.
Step 4 — Fine-tuning (the magic step) — TrOCR already knows how to read handwriting in English. Fine-tuning means you show it thousands of Odia image+text pairs and it adjusts itself to learn Odia script. You're not training from zero — you're teaching an already smart model a new script.
Step 5 — Evaluate — You test it on images it has never seen before and check how many characters it gets right. This is measured by a score called CER (Character Error Rate) — lower is better.
Step 6 — Save — You save your fine-tuned model to your PC.
Step 7 — Inference — Now you give it any new Odia handwriting image and it reads it for you.