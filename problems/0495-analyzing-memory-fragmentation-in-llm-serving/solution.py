def analyze_memory_fragmentation(block_status: list) -> dict:
    total_blocks = len(block_status)
    allocated = sum(block_status)
    total_free = total_blocks - allocated

    num_free_fragments = 0
    largest_free_fragment = 0
    current_run = 0

    for block in block_status:
        if block == 0:
            current_run += 1
        elif current_run > 0:
            num_free_fragments += 1
            largest_free_fragment = max(largest_free_fragment, current_run)
            current_run = 0

    if current_run > 0:
        num_free_fragments += 1
        largest_free_fragment = max(largest_free_fragment, current_run)

    utilization = allocated / total_blocks if total_blocks else 0.0
    fragmentation_ratio = (
        1 - largest_free_fragment / total_free if total_free else 0.0
    )

    return {
        "utilization": round(utilization, 4),
        "num_free_fragments": num_free_fragments,
        "largest_free_fragment": largest_free_fragment,
        "fragmentation_ratio": round(fragmentation_ratio, 4),
    }