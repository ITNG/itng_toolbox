
# from . import (networks,
#                drawing,
#                statistics,
#                synchrony,
#                graphUtility,
#                randomGenerators)

try:
    from . import (synchrony,
                   statistics,
                   randomGenerators,
                   graphUtility,
                   drawing,
                   networks
    )
except Exception as error:
    print (error)
    

# try:
#     from . import networks
# except Exception as error:
#     print (error)

# except:
#     print("Error")
