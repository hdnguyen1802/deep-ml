def beam_search_block_sharing(log_probs, beam_width, block_size, eos_token=-1):
    max_steps, vocab_size = log_probs.shape
    blocks = {}
    beams = []
    total_blocks_allocated = 0

    def allocate_block(tokens):
        nonlocal total_blocks_allocated
        block_id = total_blocks_allocated
        total_blocks_allocated += 1
        blocks[block_id] = {"tokens": tokens, "ref_count": 1}
        return block_id

    def release_block(block_id):
        blocks[block_id]["ref_count"] -= 1
        if blocks[block_id]["ref_count"] == 0:
            del blocks[block_id]

    if max_steps > 0:
        first_tokens = sorted(
            range(vocab_size),
            key=lambda token_id: (-log_probs[0, token_id], token_id),
        )
        for token_id in first_tokens[:beam_width]:
            beams.append({
                "score": float(log_probs[0, token_id]),
                "block_ids": [allocate_block([token_id])],
            })

    for step in range(1, max_steps):
        candidates = [
            {
                "parent_index": parent_index,
                "token_id": token_id,
                "score": beam["score"] + float(log_probs[step, token_id]),
            }
            for parent_index, beam in enumerate(beams)
            for token_id in range(vocab_size)
        ]
        candidates.sort(key=lambda candidate: (
            -candidate["score"],
            candidate["parent_index"],
            candidate["token_id"],
        ))
        selected = candidates[:beam_width]

        children_left = [0] * len(beams)
        for child in selected:
            children_left[child["parent_index"]] += 1

        for parent_index, beam in enumerate(beams):
            if children_left[parent_index] == 0:
                for block_id in beam["block_ids"]:
                    release_block(block_id)

        next_beams = []
        for child in selected:
            parent_index = child["parent_index"]
            block_ids = beams[parent_index]["block_ids"]
            children_left[parent_index] -= 1

            # The last child takes over the parent's block list.
            if children_left[parent_index] > 0:
                block_ids = block_ids.copy()
                for block_id in block_ids:
                    blocks[block_id]["ref_count"] += 1

            token_id = child["token_id"]
            last_id = block_ids[-1]
            last_block = blocks[last_id]

            if len(last_block["tokens"]) == block_size:
                block_ids.append(allocate_block([token_id]))
            elif last_block["ref_count"] > 1:
                release_block(last_id)
                block_ids[-1] = allocate_block(last_block["tokens"] + [token_id])
            else:
                last_block["tokens"].append(token_id)

            next_beams.append({"score": child["score"], "block_ids": block_ids})

        beams = next_beams

    sequences = [
        [
            token
            for block_id in beam["block_ids"]
            for token in blocks[block_id]["tokens"]
        ]
        for beam in beams
    ]
    blocks_per_beam = (max_steps + block_size - 1) // block_size

    return {
        "sequences": sequences,
        "scores": [round(beam["score"], 4) for beam in beams],
        "total_blocks_allocated": total_blocks_allocated,
        "blocks_in_use_final": len(blocks),
        "naive_blocks_needed": beam_width * blocks_per_beam,
    }