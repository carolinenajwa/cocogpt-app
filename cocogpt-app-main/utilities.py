def tool(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator