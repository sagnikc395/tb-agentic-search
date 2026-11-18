vcf files input → TB input (dataset) for each drug for big tb paper (present in unity)

right now i am thinking of using vllm to serve an llm under the hood in unity cluster and then use the biomni in a seperate process to read the inference engine as a llm provider and
use that for its tool callig.
check medea and how it is → human readbale traces (not the same as this, this uses gpt-oss-120b under the hood)

do a binary predictions for them → resistant ,not resistant, unknown  ; the agent to take in the vcf files, for given drug what will be the reason for that 

check the reasoning traces for them → for each of them -> capture the reasoning traces with vllm;
https://docs.vllm.ai/en/stable/features/reasoning_outputs/#mapping-reasoning-efforts


check their responses and how biologically valid the reasoning traces are →
