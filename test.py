# Example list
#my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#
## Function to reverse every 8-value chunk
#def reverse_chunks(lst, chunk_size):
#    return [lst[i:i + chunk_size][::-1] if i + chunk_size <= len(lst) else lst[i:][::-1] 
#            for i in range(0, len(lst), chunk_size)]
#
## Reverse every 8-value chunk
#reversed_chunks = reverse_chunks(my_list, 8)
#
## Flattening the list of reversed chunks
#flattened_list = [item for sublist in reversed_chunks for item in sublist]
#
#print(flattened_list)

#import torch
#import torch.nn.functional as F
#
## Create an example tensor `x`
#x = torch.randn(1, 5)  # For example, a tensor with shape [1, 5]
#
## Compute log softmax
#log_probs = F.log_softmax(x, dim=1)
#
## Create a mask for the values (masking the last half here)
#mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
#
## Apply the mask (setting masked log_probs to a very large negative value)
#masked_log_probs = log_probs.masked_fill(mask == 0, float('-inf'))
#
## Convert masked log probabilities to probabilities
#masked_probs = torch.exp(masked_log_probs)
#
## Renormalize the probabilities so they sum up to 1
#sum_masked_probs = masked_probs.sum(dim=1, keepdim=True)
#renormalized_probs = masked_probs / sum_masked_probs
#
## Sample from these renormalized probabilities
#sampled_index = torch.multinomial(renormalized_probs, 1)
#
#print("Log Softmax Output:", log_probs)
#print("Masked Log Softmax Output:", masked_log_probs)
#print("Renormalized Probabilities:", renormalized_probs)
#print("Sampled Index:", sampled_index.item())



#import json
#
#all_moves = []
#for i in range(64):
#    for j in range(64):
#        
#        if (i != j):
#            all_moves.append(str(i) + " " + str(j))
#
#print(len(all_moves))
#
#f = open("all_moves.txt", "w")
#f.write("\n".join(all_moves))
#f.close()
#
#f = open("all_moves.txt", "r")
#mvs = f.read().splitlines()
#print(len(mvs))
#f.close()

def square_to_number(square):
    column, row = square
    column_number = ord(column) - ord('a')
    row_number = int(row) - 1
    return row_number * 8 + column_number

def generate_numeric_moves():
    columns = 'abcdefgh'
    rows = '12345678'
    moves = set()

    for start_col in columns:
        for start_row in rows:
            start = start_col + start_row
            start_num = square_to_number(start)

            # Queen Moves
            for end_col in columns:
                for end_row in rows:
                    if end_col != start_col or end_row != start_row:
                        if end_col == start_col or end_row == start_row or abs(ord(end_col) - ord(start_col)) == abs(ord(end_row) - ord(start_row)):
                            end = end_col + end_row
                            end_num = square_to_number(end)
                            moves.add(f"{start_num} {end_num}")

            # Knight Moves
            for dx, dy in [(1, 2), (2, 1), (-1, 2), (2, -1), (-1, -2), (-2, -1), (1, -2), (-2, 1)]:
                end_col = columns.find(start_col) + dx
                end_row = rows.find(start_row) + dy
                if 0 <= end_col < 8 and 0 <= end_row < 8:
                    end = columns[end_col] + rows[end_row]
                    end_num = square_to_number(end)
                    moves.add(f"{start_num} {end_num}")

    return moves

numeric_all_moves = generate_numeric_moves()
print(len(numeric_all_moves))  # Total number of moves

f = open("all_moves.txt", "w")
f.write("\n".join(numeric_all_moves))
f.close()

f = open("all_moves.txt", "r")
mvs = f.read().splitlines()
print(len(mvs))
f.close()