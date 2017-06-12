
#data downloaded from www.fantasycruncher.com/lineup-rewind/fanduel/MLB
start<-as.Date("4-25-2017",format="%m-%d-%Y")
dates <- seq(start,as.Date("5-12-2017",format="%m-%d-%Y"),by="day")
ndates<-length(dates)

#I keep reloading df in memory, so NULL each time to start fresh
df<-NULL

df<-read.csv("04_25_2017.csv",strip.white=TRUE)

for (i in seq(1:ndates)){
  
  t<-as.character(start+(i-1))
  year<-sapply(strsplit(t,split= "-"),'[',1)
  month<-sapply(strsplit(t,split= "-"),'[',2)
  day<-sapply(strsplit(t,split= "-"),'[',3)
  
  file<-paste0(month,"_",day,"_",year,".csv")
  tempdf<-read.csv(file,strip.white=TRUE)
  df<-rbind(df,tempdf)
}


#park factors
park<-read.csv("Park_Factors.csv",header=TRUE)

#attach a park factor to each entry
park_factor<-rep(0,nrow(df))
for(i in 1:nrow(df)){
 if(as.character(df[i,"Opp"])=="@"){
   park_factor[i]<-as.numeric(as.character(park[df[i,"OppS"],"X.2"]))
  }
  else{
   park_factor[i]<-as.numeric(as.character(park[df[i,"Team"],"X.2"]))
  }
}
park_factor[is.na(park_factor)] <- 0


#Handness

match_hand<-rep(0,nrow(df))
#When the batter and pitcher are same handed = 1, else 0
for(i in 1:nrow(df)){
  if(as.character(df[i,'Hand'])==as.character(df[i,'Pitcher.H'])){
    match_hand[i]<-1
  }else{
    match_hand[i]<-0
  }
  
}

#query espn for latest fielding percentages by team
#fielding_espn<-"http://www.espn.com/mlb/stats/team/_/stat/fielding"
#fielding<-readHTMLTable(fielding_espn,stringsAsFactors=FALSE)

city_names<-read.csv("city_names.csv",header=FALSE,strip.white=TRUE)
#I don't know why, but the readHTMLTable reads a df of NULL dim.  Fix later

fielding<-read.csv('field_5_1.csv',strip.white=TRUE)
field_per<-rep(0,nrow(df))
for(i in 1:nrow(df)){
  #espn lists field perct by full city name, not abbr. (ie 'Boston' NOT 'BOS')
  #so convert
  city<-as.character(city_names[city_names$V1==as.character(df[i,'OppS']),2])
  field_per[i]<-fielding[fielding$TEAM==city,'FPCT']
}

#pitching data from espn
#figure out what's wrong with the scrape above
#then rewrite to scrape from: http://www.espn.com/mlb/stats/team/_/stat/pitching/type/expanded-2

pitching<-read.csv('pitching_5_1.csv',strip.white=TRUE)
team_WHIP<-rep(0,nrow(df))

pitch_data<-function(var){
  return_var<-rep(0,nrow(df))
  for(i in 1:nrow(df)){
    #espn lists field perct by full city name, not abbr. (ie 'Boston' NOT 'BOS')
    #so convert
    city<-as.character(city_names[city_names$V1==as.character(df[i,'OppS']),2])
    return_var[i]<-pitching[pitching$TEAM==city,var]
  }
  return(return_var)
}
team_WHIP<-pitch_data("WHIP")
#Scale the next pitching features
team_KBB<-pitch_data("K.BB")
team_KBB<-team_KBB/mean(team_KBB)
team_K9<-pitch_data("K.9")
team_K9<-team_K9/mean(team_K9)
team_DIPS<-pitch_data("DIP.")
team_DIPS<-team_DIPS/mean(team_DIPS)


## ASSEMBLE FEATURE SET

df_features<-df[,'ISO']
#code treats "Actual.Score" as the output
df_features<-cbind(df_features,df$wOBA,field_per,df$Actual.Score)
names(df_features)[1:4]<-c('ISO','wOBA','BAvg','OppERA')

#Create random numbers as a baseline to check model efficacy
df_random<-cbind(runif(nrow(df)),runif(nrow(df)),df$Actual.Score)
colnames(df_random)<-c("X","Y","Z")

#replace NAs with 0s to make Caret happy
df_features[is.na(df_features)] <- 0
df_random[is.na(df_random)] <- 0


install.packages('caret',dependencies = TRUE)
library(caret)
install.packages('glmnet')
library(glmnet)
library(Matrix)
install.packages('Matrix')

#to create a learning curve, I will work with subsets of the training set:
len <- c(20,125,200,450,500,750,1200,1700,2500,3500,4200)
error<-rep(0,length(len))
error2<-rep(0,length(len))
error3<-rep(0,length(len))
CV_error<-rep(0,length(len))

for(i in 1:length(len)){

  #loop over various data set sizes
  #first create a training, testing, and cross validation set from available data

  df_new <- df_features[1:len[i],]

  inTraining <- createDataPartition(df_new[,1], p = .6, list = FALSE)
  training <- df_new[ inTraining,1:(ncol(df_new)-1)]
 
  temp <- df_new[-inTraining,1:(ncol(df_new))]
  
  CVsplit <- createDataPartition(temp[,ncol(df_new)],p=0.5,list= FALSE)

  testing  <- temp[-CVsplit,1:(ncol(df_new)-1)]
  CV <- temp[CVsplit,1:(ncol(df_new)-1)]

  outcomes <- df_new[inTraining,ncol(df_new)]
  testOutcomes <-temp[-CVsplit,ncol(df_new)]
  crossValOut <- temp[CVsplit,ncol(df_new)]

  train_control <-trainControl(method="cv",number=10)
  #Regular regression
  fit <- train(x=training,y=outcomes,method='lm',trControl = train_control)

  p<-predict(fit,testing)
  
  CVp<-predict(fit,CV)

  #compare these predictions against
  error[i]<-RMSE(p,testOutcomes)
  CV_error[i]<-RMSE(p,crossValOut)
  
  #Now try modeling with a penalized regression model (L1)
#  fit2 <- train(x=training,y=outcomes,method='blassoAveraged',trControl = train_control)
  
#  p2<-predict(fit2,testing)
#  error2[i]<-RMSE(p2,testOutcomes)
  
  #And L2
#  fit3 <- train(x=training,y=outcomes,method='bridge',trControl = train_control)
  
#  p3<-predict(fit3,testing)
#  error3[i]<-RMSE(p3,testOutcomes)
  
}
plot(x=len,y=error,xlab="Data set size",ylab="Model Error",col="blue",pch=14,ylim=c(1,15),main="Non-Penalized Regression")
legend('topright', c("Test Set","Cross Val Set"),lty=1, lwd=4,bty='n', cex=.75,col=c("blue","red"))
points(x=len,y=CV_error,xlab="Data set size",ylab="Model Error",col="red",pch=16)

plot(x=len,y=error,xlab="Data set size",ylab="Model Error",col="blue",pch=14,ylim=c(1,15),main="Regression Models")
legend('topright', c("Non-Penalized","Bayesian Ridge Regression","L2"),lty=1, lwd=4,bty='n', cex=.75,col=c("blue","red","green"))
points(x=len,y=error2,xlab="Data set size",ylab="Model Error",col="red",pch=16)
points(x=len,y=error3,xlab="Data set size",ylab="Model Error",col="green",pch=16)


#more data doesnt seem to improve performance, let's try some PCA
#dont ask me why, but preprocess doesnt like the feature names
names<-c(1,2,3,4,5,6,7,8,9,10,11,12)
df_PCA<-df_features[,-13]
colnames(df_PCA)<-names
df.pca<-preProcess(df_PCA,method="pca")
df.pca2<-prcomp(df_PCA)
summary(df.pca2)

cor(df_PCA)
