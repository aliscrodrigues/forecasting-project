import torch

def train_model(model, train_loader, val_loader, epochs=10):
    criterion = torch.nn.MSELoss() 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train() 
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()                  
            predictions = model(X_batch)           
            loss = criterion(predictions, y_batch) 
            loss.backward()                       
            optimizer.step()                       

        model.eval() 
        val_loss = 0
        with torch.no_grad(): 
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()
        
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")
    
    return model