# -*- coding: utf_8 -*-
import direct.directbase.DirectStart 
from pandac.PandaModules import *
from direct.showbase.DirectObject import DirectObject
from direct.directtools.DirectSelection import DirectBoundingBox

#Определим класс, который позволит загружать геометрию из файла и добавлять этой 
#геометрии физику ODE

class physBox(object):
    def __init__(self,world,space,NodePath):
        self.Name = NodePath.getName()
        
        self.boxNP = NodePath.copyTo(render)
        self.boxNP.setPos(NodePath.getPos(render))
        self.boxNP.setHpr(NodePath.getHpr(render))
        self.boxNP.setColor(1, 1, 0, 1)
        #self.boxNP.flattenStrong()
        
        self.body = OdeBody(world)
        M = OdeMass()
        bb=DirectBoundingBox(self.boxNP) 

        dimX = (float(bb.getMax().getX()) - float(bb.getMin().getX()))
        dimY = (float(bb.getMax().getY()) - float(bb.getMin().getY()))
        dimZ = (float(bb.getMax().getZ()) - float(bb.getMin().getZ()))

        bb.show()
        M.setBox(100, dimX,dimY,dimZ)
        self.body.setMass(M)
        self.body.setPosition(self.boxNP.getPos(render))
        self.body.setQuaternion(self.boxNP.getQuat(render))
        
        boxGeom = OdeBoxGeom(space, dimX, dimY, dimZ)
        boxGeom.setBody(self.body)
    
    def set_Pos(self,x,y,z):
        self.boxNP.setPos(x,y,z)
        self.body.setPosition(self.boxNP.getPos(render))
    
    def set_Hpr(self,x,y,z):
        self.boxNP.setHpr(x,y,z)
        self.body.setQuaternion(self.boxNP.getQuat(render))
        
    def get_Body(self):
        return self.body
        
    def progress(self):
        self.boxNP.setPos(render, self.body.getPosition())
        self.boxNP.setQuat(render,Quat(self.body.getQuaternion()))

#Определим класс который будет соединять загружаемые physBox'ы посредством 
#Joint соединений
class RagDoll(object):
    
    def __init__(self,pos,hpr):
        
        self.BoneList = []
        base.disableMouse()
        base.camera.setPosHpr(0,-55,-5,0,0,0)
        
        self.NPMap = loader.loadModel('models/Map.egg')
        self.NPMap.reparentTo(render)
        self.NPMap.setZ(-12)
        self.NPMap.flattenStrong()
        
        #Загружаем файл с данными о геометрич. объектах и их соединениях 
        self.rdD = loader.loadModel('models/rdData.egg')
        self.rdD.setPos(pos[0],pos[1],pos[2])
        self.rdD.setHpr(hpr[0],hpr[1],hpr[2])
        
        self.world = OdeWorld()
        self.world.setGravity(0, 0, -9.81)
        
        self.world.initSurfaceTable(1)
        self.world.setSurfaceEntry(0, 0, 0.8, 0.0, 10    , 0.9, 0.00001, 100, 0.002)
        
        self.space = OdeSimpleSpace()
        self.space.setAutoCollideWorld(self.world)
        self.contactgroup = OdeJointGroup()
        self.space.setAutoCollideJointGroup(self.contactgroup)
        
        # Выполняем инициализацию геометрии с физикой
        self.CreateRDBones(self.rdD)
        
        #Создаем соответствующие соединения
        self.CreateRDJoints(self.rdD)
        
        self.MapTrimesh = OdeTriMeshData(self.NPMap, False)
        self.MapGeom = OdeTriMeshGeom(self.space, self.MapTrimesh)
        self.MapGeom.setCollideBits(BitMask32(0x00000002))
        self.MapGeom.setCategoryBits(BitMask32.allOff())
          
        taskMgr.doMethodLater(0.5, self.simulationTask, "Physics Simulation")
        

    def simulationTask(self,task):
        
        self.space.autoCollide() 
        self.world.quickStep(globalClock.getDt()) 
        if (len(self.BoneList)>0): 
            for i in (self.BoneList):
                
                i.progress()
                
        self.contactgroup.empty()
        return task.cont
    
    def CreateRDBones(self,rdData):
        rdBones = rdData.findAllMatches('**/bone*')
        numBones = rdBones.getNumPaths()
        for i in range(numBones):
            bone = physBox(self.world,self.space,rdBones[i])
            self.BoneList.append(bone)
    
    def CreateRDJoints(self,rdData):
        rdHJoints = rdData.findAllMatches('**/hingeJoint*')
        numHJoints = rdHJoints.getNumPaths()
        if (numHJoints>0):
            for j in range(numHJoints):
                Pbody = None
                Cbody = None
                for b in (self.BoneList):
                    
                    if (b.Name == rdHJoints[j].getTag('Jwith')):
                        Pbody = b.get_Body()
                        
                    if (b.Name == rdHJoints[j].getTag('Jthis')):
                        Cbody = b.get_Body()
                        
                if (Pbody != None) and (Cbody != None):
                    
                    hj=OdeHingeJoint(self.world)
                    #hj.setParamHiStop(0.5)
                    #hj.setParamLoStop(-0.5)
                    hj.attachBodies(Cbody,Pbody)
                    hj.setAnchor(rdHJoints[j].getX(render), rdHJoints[j].getY(render), rdHJoints[j].getZ(render))
                    hj.setAxis(rdHJoints[j].getQuat(render).getRight())
                    
                    
        rdBJoints = rdData.findAllMatches('**/ballJoint*')
        numBJoints = rdBJoints.getNumPaths()
        if (numBJoints>0):
            for j in range(numBJoints):
                Pbody = None
                Cbody = None
                for b in (self.BoneList):
                    
                    if (b.Name == rdBJoints[j].getTag('Jwith')):
                        Pbody = b.get_Body()
                        
                    if (b.Name == rdBJoints[j].getTag('Jthis')):
                        Cbody = b.get_Body()
                        
                if (Pbody != None) and (Cbody != None):
                    
                    bj=OdeBallJoint(self.world)
                    bj.attachBodies(Cbody,Pbody)
                    bj.setAnchor(rdBJoints[j].getX(render), rdBJoints[j].getY(render), rdBJoints[j].getZ(render))


rd=RagDoll((5,0,20),(0,60,-60))
run()