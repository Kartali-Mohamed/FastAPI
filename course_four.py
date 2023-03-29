from fastapi import FastAPI,  HTTPException #added
import mysql.connector

#added
from pydantic import BaseModel

#added
class Student(BaseModel):
    name: str
    age: int
    department: str
    
    
    
dbcon = mysql.connector.connect(
    host="localhost",
    user="muhammed",
    password="muhammed",
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
    if len(names) == 0: #added
        raise HTTPException(status_code=500, detail="Student not found")
    return names[0] 

@app.post("/names/create")
async def create_name(student:Student): #added Student
    sql = "INSERT INTO `names`(`name`, `age`, `department`) VALUES (%s,%s,%s)"
    val = (student.name,student.age,student.department)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"message": "name  has been created successfully" } 


@app.post("/names/update")
async def update_name(my_id:int,student:Student): #added Student
    sql ="UPDATE `names` SET  `name` = %s , `age` = %s , `department` = %s WHERE id = %s"
    val = (student.name,student.age,student.department,my_id )
    mycursor.execute(sql, val)
    dbcon.commit()
    return student


@app.delete("/names/{id}")
async def delete_name(myid:int):
    sql ="DELETE FROM `names` WHERE id = %s"
    val = (myid,)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"message": "name  has been deleted successfully"} 














