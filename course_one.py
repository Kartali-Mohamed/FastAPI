from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"Hello": "World"}

names = [{ "name":"muhammed","age":39,"department":"IT"},
        { "name":"ahmed","age":31,"department":"HR"},
        { "name":"osama","age":42,"department":"Finance"},
        { "name":"ali","age":45,"department":"AI"}]

@app.get("/names")
def get_names():
    return names 

@app.get("/names/{id}")
def get_names_by_id(id:int):
    return names[id] 

@app.delete("/names/{id}")
def delete_name(id:int):
    names.pop(id)
    return {"message": "name  has been deleted successfully",
           "name":names[id],
           "names":names} 


@app.post("/names/create")
def create_name(name:dict):
    names.append(name)
    return {"message": "name  has been created successfully",
           "added":name,
           "names":names } 



@app.post("/names/update")
def update_name(id:int,name:dict):
    name_update = names[id]
    name_update['name'] = name['name']
    name_update['age'] = name['age']
    name_update['department'] = name['department']
    names[id] = name_update
     
    return {"message": "name has been updated successfully",
           "name_update":name_update,
           "names":names } 


if __name__ == "__main__":
    uvicorn.run("course_one:app", host='127.0.0.1', port=8000, reload=True)






