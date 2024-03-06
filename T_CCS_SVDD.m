function EVAL = T_CCS_SVDD(Traindata,Trainlabel,Testdata,Testlabel,C,beta,delta,eta,d,GPU,numofmodes)  
    warning("off");    
    %numofmodes=2; 
   miu1=1;     
   miu2=1;    
    %d=2;  
    maxIter=6; 
    %eta=0.1;
    %C=0.3;      %越小，圈越大
    %beta=1;       
    %delta=0.5;
    
    [Q,svdd]=MainFunction(Traindata,Trainlabel,d,numofmodes,maxIter,C,beta,delta,miu1,miu2,eta,GPU);
    EVAL=Evaluate(Q,Testdata,Testlabel,svdd,numofmodes);
    %PlotData(Q,Traindata,R,center,Testdata)
end
