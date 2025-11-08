
algorithm_name=dlc-nov07-meg

region=ap-southeast-3
account=$(aws sts get-caller-identity --query Account --output text)

aws ecr describe-repositories --region $region --repository-names "${algorithm_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
echo "create repository:" "${algorithm_name}"
aws ecr create-repository --region $region  --repository-name "${algorithm_name}" > /dev/null
fi

aws ecr get-login-password --region ${region}|docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"

docker build -t ${algorithm_name} -f dockerfile.dlc.nov7 .

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
docker tag ${algorithm_name} ${fullname}
docker push ${fullname}

echo $fullname