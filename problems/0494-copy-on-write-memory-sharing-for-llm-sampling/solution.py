def cow_memory_simulator(
    prompt_length: int,
    num_sequences: int,
    gen_lengths: list,
    block_size: int,
) -> dict:
    shared_prompt_blocks, prompt_remainder = divmod(prompt_length, block_size)
    has_partial_block = prompt_remainder > 0
    spare_slots = block_size - prompt_remainder if has_partial_block else 0

    references = num_sequences
    cow_events = 0
    new_generation_blocks = 0
    total_logical_blocks = 0

    for gen_length in gen_lengths:
        total_tokens = prompt_length + gen_length
        total_logical_blocks += (total_tokens + block_size - 1) // block_size

        if has_partial_block and gen_length > 0 and references > 1:
            cow_events += 1
            references -= 1

        remaining_tokens = max(0, gen_length - spare_slots)
        new_generation_blocks += (remaining_tokens + block_size - 1) // block_size

    original_prompt_blocks = shared_prompt_blocks + int(has_partial_block)
    total_physical_blocks = original_prompt_blocks + cow_events + new_generation_blocks

    memory_saved_pct = 0.0
    if total_logical_blocks > 0:
        memory_saved_pct = 100 * (1 - total_physical_blocks / total_logical_blocks)

    return {
        "total_physical_blocks": total_physical_blocks,
        "total_logical_blocks": total_logical_blocks,
        "shared_prompt_blocks": shared_prompt_blocks,
        "cow_events": cow_events,
        "new_generation_blocks": new_generation_blocks,
        "memory_saved_pct": round(memory_saved_pct, 2),
    }