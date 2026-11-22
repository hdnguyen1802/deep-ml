import numpy as np


class VirtualKVCacheManager:
    def __init__(self, num_physical_blocks: int, block_size: int, hidden_dim: int):
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        shape = (num_physical_blocks, block_size, hidden_dim)
        self.keys = np.zeros(shape)
        self.values = np.zeros(shape)
        self.free_blocks = set(range(num_physical_blocks))
        self.page_tables = {}
        self.next_positions = {}

    def allocate_block(self, seq_id: int) -> int:
        if not self.free_blocks:
            return -1

        physical_id = min(self.free_blocks)
        self.free_blocks.remove(physical_id)
        page_table = self.page_tables.setdefault(seq_id, [])
        page_table.append(physical_id)
        return len(page_table) - 1

    def write(self, seq_id: int, position: int, key, value) -> bool:
        logical_block, slot = divmod(position, self.block_size)
        allocated_blocks = len(self.page_tables.get(seq_id, []))
        blocks_needed = max(0, logical_block + 1 - allocated_blocks)

        if blocks_needed > len(self.free_blocks):
            return False

        for _ in range(blocks_needed):
            self.allocate_block(seq_id)

        physical_id = self.page_tables[seq_id][logical_block]
        self.keys[physical_id, slot] = key
        self.values[physical_id, slot] = value
        self.next_positions[seq_id] = max(
            self.next_positions.get(seq_id, 0), position + 1
        )
        return True

    def append(self, seq_id: int, key, value) -> bool:
        position = self.next_positions.get(seq_id, 0)
        return self.write(seq_id, position, key, value)

    def read(self, seq_id: int, position: int):
        logical_block, slot = divmod(position, self.block_size)
        page_table = self.page_tables.get(seq_id, [])

        if 0 <= logical_block < len(page_table):
            physical_id = page_table[logical_block]
            return (
                self.keys[physical_id, slot].copy(),
                self.values[physical_id, slot].copy(),
            )

        return None, None

    def free_sequence(self, seq_id: int) -> int:
        page_table = self.page_tables.pop(seq_id, [])
        self.next_positions.pop(seq_id, None)

        for physical_id in page_table:
            self.keys[physical_id].fill(0)
            self.values[physical_id].fill(0)

        self.free_blocks.update(page_table)
        return len(page_table)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_page_table(self, seq_id: int) -> list:
        return self.page_tables.get(seq_id, []).copy()