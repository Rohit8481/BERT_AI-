from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "my_email_phishing_detector"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device("cpu")
model.to(device)
model.eval()

############### taking the input from user ##################

lines = []
print("Enter text (type END to finish):")

while True:
    line = input()
    if line == "END":
        break
    lines.append(line)

data = "\n".join(lines)
print("\nWait calculating.....")
###################### tokenizaion ########################
tok = tokenizer(data, padding=True, truncation=True, max_length=128, return_tensors= 'pt')

tok['input_ids'] = tok['input_ids'].to(device)              # send all the tokens to the device such as CPU or GPU, in my case i only use CPU to process
tok['attention_mask'] = tok['attention_mask'].to(device)    # send all the tokens to the device such as CPU or GPU, in my case i only use CPU to process


with torch.no_grad():
    output = model(**tok)
    logits = output.logits
    x = logits[0][0].item()
    y = logits[0][1].item()
    
################# Softmax ( formulae : prob_x = (e^x / e^x + e^y)*100 )#######################
    
softmax_1 = 2.71828182845904523536028747135266249775724709369995 ** x # e^x  
softmax_2 = 2.71828182845904523536028747135266249775724709369995 ** y # e^y
sum = softmax_1 + softmax_2 # e^x + e^y 

final_1 = (softmax_1 / sum)*100
final_2 = (softmax_2 / sum)*100


##############################################################################
def iden(final_1, final_2):
    if final_1 < final_2 : return True
    else : return False

prob = iden(final_1,final_2)    

if prob : print("\nPhishing ❌\n") 
else : print("\nlegitimate ✅\n")

