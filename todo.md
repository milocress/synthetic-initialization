# Milestones

(description)

- (list of things we need to get done)

Status: (status)

Blockers: (list of things we have to de before we can do this milestone)

## Synthetic Identity Initialization

Handcraft weights of T5 model to perform the identity task out of the box.

- identified a bug that blocks benchmarks on this model

Status: complete but buggy

## Synthetic Set Initialization

Handcraft weights of T5 model to perform the set deduplication task out of the box.

- needs layer to replace nonduplicate characters with `<pad>`

Status: mostly complete

## Finetune benchmarks

### identity (pretrained)

- Use results from paper to compare to downstream performance of the synthetic initialized identity.

status: complete (from paper)

### set (pretrained)

- Use results from paper to compare to downstream performance of the synthetic initialized set.

status: complete (from paper)

### baseline (random initialization)

- Run finetuning on model with random initializations.
- Build a script for finetuning that can take an arbitrary model and return a model optimized for a given downstream task.
- Get training results and compile them to a graph.
- status: not started
- needs finetuning script for huggingface transformers and downstream tasks

blockers: none (this is ready and needs to be done soon)

### identity (initialized)

- Run finetuning on model with identity synthetic initialization.
- Use the finetuning script from (baseline).
- Get training results and compile them to a graph.

status: not started

blockers:

- baseline
- bug fix for decoder

### set (initialized)

status: not started

blockers:

- baseline
- synthetic nonduplicate padding layer

## Generalized Synthetic Weight Generation

status: todo, fingers crossed

## Slides for Final Presentation

status: todo, very important!

- we have been painting very localized pictures in our two demos
- for our final demo we need to clearly restate our goals and our line of thinking
- then we need to ideally show the results
- as a stretch goal it would be nice to demostrate the generalized synthetic weight generation
