import torch
import os
import matplotlib.pyplot as plt

def train(model, train_data, val_data, optimizer, criterion, device, epochs=15, is_rnn = False):
    # print("in rnn train")
    ta = []
    va = []
    tl = []
    vl = []
    # try:
    for epoch in range(epochs):
        model.train()
        tloss = 0
        tacc = 0
        totsamps = 0
        for batch in train_data:
            ip, mask, op = batch
            ip = ip.to(device) 
            op = op.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(ip, mask)
            loss = criterion(output, op)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            tacc += (output.argmax(1) == op).sum().item()
            totsamps += len(op)
        #   ta = tacc / totsamps    
        val_loss, val_acc = evaluate(model, val_data, criterion, device)
        ta.append(tacc/totsamps)
        va.append(val_acc)
        tl.append(tloss/len(train_data))
        vl.append(val_loss)

        print(f"Epoch {epoch+1}: Training loss: {(tloss/len(train_data)):.5f}, Training accuracy: {(tacc/totsamps):5f}, \n"
        f"Validation loss: {(val_loss):.5f}, Validation accurracy: {(val_acc):.5f}")

    # except Exception as e:
    # print(f"An error occurred: {e}")
    model = 'rnn' if is_rnn else 'LoRA'
    print("Plotting...")
    plot_accuracy(ta, va, model)
    plot_loss(tl, va, model)
    print("Plots saved as ", model,"_accuracy.png and ", model,"_loss.png",)
#     plt.plot(tl, label='Train Loss')
#     plt.plot(vl, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.plot(ta, label='Train Accuracy')
#     plt.plot(va, label='Val Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.close()



def train_distil(teacher_model, student_model, train_data, val_data, optimizer, criterion, distil_criterion, device, epochs=15, alpha=0.8):
    # print("in distil")
    # ta, va,tl,vl = [[]]*4
    ta = []
    va = []
    tl = []
    vl = []
    # val_accs = []
    # train_losses = []
    # val_losses = []
    for epoch in range(epochs):
        student_model.train()
        tloss = 0
        tacc = 0
        totsamps = 0
        for batch in train_data:
            ip, mask, op = batch
            ip = ip.to(device) 
            op = op.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = student_model(ip, mask)
            teacher_output = teacher_model(ip, mask)
            teacher_output = torch.nn.functional.softmax(teacher_output, dim=1)
            student_loss = criterion(output, op)
            distillation_loss = distil_criterion(output, teacher_output)
#           loss = alpha * student_loss + (1 - alpha) * distillation_loss
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            tacc += (output.argmax(1) == op).sum().item()
            totsamps += len(op)
        #     val_loss, val_acc = evaluate(student_model, val_data, student_criterion, device)
        val_loss, val_acc = evaluate(student_model, val_data, criterion, device)
        ta.append(tacc/totsamps)
        va.append(val_acc)
        tl.append(tloss/len(train_data))
        vl.append(val_loss)
        print(f"Epoch {epoch+1}: Training loss: {(tloss/len(train_data)):.5f}, Training accuracy: {(tacc/totsamps):5f}, \n"
        f"Validation loss: {(val_loss):.5f}, Validation accurracy: {(val_acc):.5f}")
    print("Plotting...")
    plot_accuracy(ta, va, 'Distillation')
    plot_loss(tl, vl, 'Distillation')
    print("Plots saved as Distillation_accuracy.png and Distillation_loss.png")
#     plt.plot(tl, label='Train Loss')
#     plt.plot(vl, label='Val Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.plot(ta, label='Train Accuracy')
#     plt.plot(va, label='Val Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.close()
    
def evaluate(model, data, criterion, device):
    model.eval()
    tl= 0
    acc = 0
    dlen = 0
    with torch.no_grad():
        for batch in data:
            ip, mask, op = batch
            ip = ip.to(device) 
            op = op.to(device)
            mask = mask.to(device)
            output = model(ip, mask)
            loss = criterion(output, op)
            acc += (output.argmax(1) == op).sum().item()
            dlen += len(op)
            tl += loss.item()
    return tl / len(data), acc / dlen
    
def plot_accuracy(train_acc, val_acc, model_name):
    plt.figure()
    x = range(len(train_acc))
    plt.plot(x, train_acc, label='Training Accuracy')
    plt.plot(x, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    ylabel = f"{model_name} Accuracy"
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{model_name} Accuracy Over Epochs", fontsize=14)  
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left')
    # plt.savefig(model_name + '_accuracy.png', dpi=300)
    plt.savefig(os.path.join("plots", model_name + '_accuracy.png'), dpi=300)
    # plt.savefig(model_name + '_accuracy.png')
    print("Plotting accuracy done")
    plt.close()
    print("Saving...")

def plot_loss(train_loss, val_loss, model_name):
    plt.figure()
    x = range(len(train_loss))
    plt.plot(x, train_loss, label='Training Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    ylabel = f"{model_name} Loss"
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{model_name} Losses Over Epochs", fontsize=14)  
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left')
    # plt.savefig(model_name + '_loss.png', dpi=300)
    # plt.savefig(model_name + '_loss.png')
    plt.savefig(os.path.join("plots", model_name + '_loss.png'), dpi=300)
    print("Plotting losses done")
    plt.close()
    print("Saving...")
