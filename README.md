# ML
trainingWmomentum.py
    #TODO: 
        make a function for the momentum that run from 0 - 0.9
        find the average of the results
        use the average to run the final time
        display the results in graph

show multiple images in one figure:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

