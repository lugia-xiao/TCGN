def find_sample_number(sample_name):
    sample_list = ["A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "C6",
                   "D1", "D2", "D3", "D4", "D5", "D6", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3"]
    for i in range(len(sample_list)):
        if sample_list[i]==sample_name:
            return i

if __name__=="__main__":
    sample_list = ["A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "C6",
                   "D1", "D2", "D3", "D4", "D5", "D6", "E1", "E2", "E3", "F1", "F2", "F3", "G1", "G2", "G3"]
    for sample in sample_list:
        print(sample,find_sample_number(sample))