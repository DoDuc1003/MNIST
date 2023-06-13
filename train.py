import torch
def write_to_file(file_path, content):
    
    with open(file_path, 'a') as file:
        file.write(content)

def train_epoch(model, train_dataloader, loss_func, optimizer, epoch):
    model.train()
    total_loss = 0
    for X, y in train_dataloader:
        model.cuda()
        X = X.view(len(X), -1).cuda()
        out = model(X)
        
        loss = loss_func(out, y.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (len(train_dataloader) * 1.0)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    write_to_file("result/result.txt", f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    

def test_epoch(model, test_dataloader, loss_func, optimizer, epoch):
    model.eval()
    model.to("cpu")
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.view(len(inputs), -1)
            outputs = model(inputs)  # Forward pass
            loss = loss_func(outputs, labels)

            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(test_dataloader)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")
    write_to_file("result/result.txt", f"Test Accuracy: {accuracy:.4f}")
    write_to_file("result/result.txt", f"Test Loss: {avg_loss:.4f}")
        