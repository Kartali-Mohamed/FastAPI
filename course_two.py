from fastapi import FastAPI
import mysql.connector
import uvicorn

dbcon = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="pythondb")

mycursor = dbcon.cursor()
app = FastAPI()


@app.get("/")
async def root():
    return {"Hello": "World"}

@app.get("/names")
async def get_names():
    sql = "SELECT * FROM names"
    mycursor.execute(sql)
    names = mycursor.fetchall()
    return names 

@app.get("/names/{myid}")
async def get_names_by_id(myid:int):
    sql = "SELECT * FROM names WHERE id = %s"
    val = (myid,)
    mycursor.execute(sql,val)
    names = mycursor.fetchall()
    return names[0] 

@app.post("/names/create")
async def create_name(name:dict): 
    sql = "INSERT INTO `names`(`name`, `age`, `department`) VALUES (%s,%s,%s)"
    val = (name['name'],name['age'],name['department'])
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"message": "name  has been created successfully" } 


@app.post("/names/update")
async def update_name(my_id:int,name:dict): 
    sql ="UPDATE `names` SET  `name` = %s , `age` = %s , `department` = %s WHERE id = %s"
    val = (name['name'],name['age'],name['department'],my_id )
    mycursor.execute(sql, val)
    dbcon.commit()
    return {"message": "name has been update successfully"}


@app.delete("/names/{id}")
async def delete_name(myid:int):
    sql ="DELETE FROM `names` WHERE id = %s"
    val = (myid,)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"message": "name  has been deleted successfully"} 


if __name__ == "__main__":
    uvicorn.run("course_two:app", host='127.0.0.1', port=8000, reload=True)











