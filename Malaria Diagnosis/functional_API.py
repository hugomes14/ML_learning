#At this moment, I have working with the sequential API. However, there are the functional API and the subclassing API. One limitation
#of the sequential API is that is not possible to, for example, besides to determinate if the cell is parasitized or not, 
#also determine its position in the image. This is possible with the functional API.

#The functional API is more flexible than the sequential API. It allows to create more complex models, 
#such as multi-output models, directed acyclic graphs, or models with shared layers.

#The subclassing API provides the most flexibility, but is also more complex and can be more difficult to debug. It is based 
# on object-oriented programming. Where the model is a class, and the layers are attributes of the class.