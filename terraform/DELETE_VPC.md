# How to Delete Existing VPCs

Before deleting VPCs, you need to delete all resources inside them first.

## Step 1: List Your VPCs

```bash
aws ec2 describe-vpcs --query 'Vpcs[*].[VpcId,CidrBlock,Tags[?Key==`Name`].Value|[0]]' --output table
```

## Step 2: Delete Resources in Each VPC

For each VPC you want to delete, you need to remove:

### 2.1 Delete Internet Gateways

```bash
# List IGWs
aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=vpc-xxxxx"

# Detach and delete
aws ec2 detach-internet-gateway --internet-gateway-id igw-xxxxx --vpc-id vpc-xxxxx
aws ec2 delete-internet-gateway --internet-gateway-id igw-xxxxx
```

### 2.2 Delete NAT Gateways

```bash
# List NAT Gateways
aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=vpc-xxxxx"

# Delete (takes a few minutes)
aws ec2 delete-nat-gateway --nat-gateway-id nat-xxxxx
```

### 2.3 Delete Subnets

```bash
# List subnets
aws ec2 describe-subnets --filters "Name=vpc-id,Values=vpc-xxxxx"

# Delete each subnet
aws ec2 delete-subnet --subnet-id subnet-xxxxx
```

### 2.4 Delete Route Tables (except main)

```bash
# List route tables
aws ec2 describe-route-tables --filters "Name=vpc-id,Values=vpc-xxxxx"

# Delete custom route tables (not the main one)
aws ec2 delete-route-table --route-table-id rtb-xxxxx
```

### 2.5 Delete Security Groups (except default)

```bash
# List security groups
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=vpc-xxxxx"

# Delete (can't delete default SG)
aws ec2 delete-security-group --group-id sg-xxxxx
```

### 2.6 Delete Network ACLs (except default)

```bash
# List NACLs
aws ec2 describe-network-acls --filters "Name=vpc-id,Values=vpc-xxxxx"

# Delete custom NACLs
aws ec2 delete-network-acl --network-acl-id acl-xxxxx
```

### 2.7 Delete VPC Endpoints

```bash
# List endpoints
aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=vpc-xxxxx"

# Delete
aws ec2 delete-vpc-endpoint --vpc-endpoint-id vpce-xxxxx
```

## Step 3: Delete the VPC

Once all resources are deleted:

```bash
aws ec2 delete-vpc --vpc-id vpc-xxxxx
```

## Automated Script

Here's a script to help delete a VPC (use with caution!):

```bash
#!/bin/bash
VPC_ID=$1

if [ -z "$VPC_ID" ]; then
  echo "Usage: $0 <vpc-id>"
  exit 1
fi

echo "Deleting resources in VPC: $VPC_ID"

# Delete Internet Gateways
for IGW in $(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC_ID" --query 'InternetGateways[*].InternetGatewayId' --output text); do
  echo "Detaching and deleting IGW: $IGW"
  aws ec2 detach-internet-gateway --internet-gateway-id $IGW --vpc-id $VPC_ID
  aws ec2 delete-internet-gateway --internet-gateway-id $IGW
done

# Delete NAT Gateways
for NAT in $(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$VPC_ID" --query 'NatGateways[?State==`available`].NatGatewayId' --output text); do
  echo "Deleting NAT Gateway: $NAT"
  aws ec2 delete-nat-gateway --nat-gateway-id $NAT
  echo "Waiting for NAT Gateway to be deleted..."
  aws ec2 wait nat-gateway-deleted --nat-gateway-ids $NAT
done

# Delete VPC Endpoints
for ENDPOINT in $(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$VPC_ID" --query 'VpcEndpoints[*].VpcEndpointId' --output text); do
  echo "Deleting VPC Endpoint: $ENDPOINT"
  aws ec2 delete-vpc-endpoint --vpc-endpoint-id $ENDPOINT
done

# Delete Subnets
for SUBNET in $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query 'Subnets[*].SubnetId' --output text); do
  echo "Deleting Subnet: $SUBNET"
  aws ec2 delete-subnet --subnet-id $SUBNET
done

# Delete Route Tables (except main)
for RT in $(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC_ID" --query 'RouteTables[?Associations[0].Main!=`true`].RouteTableId' --output text); do
  echo "Deleting Route Table: $RT"
  aws ec2 delete-route-table --route-table-id $RT
done

# Delete Security Groups (except default)
for SG in $(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=!default" --query 'SecurityGroups[*].GroupId' --output text); do
  echo "Deleting Security Group: $SG"
  aws ec2 delete-security-group --group-id $SG
done

# Delete Network ACLs (except default)
for ACL in $(aws ec2 describe-network-acls --filters "Name=vpc-id,Values=$VPC_ID" "Name=default,Values=false" --query 'NetworkAcls[*].NetworkAclId' --output text); do
  echo "Deleting Network ACL: $ACL"
  aws ec2 delete-network-acl --network-acl-id $ACL
done

# Finally, delete the VPC
echo "Deleting VPC: $VPC_ID"
aws ec2 delete-vpc --vpc-id $VPC_ID

echo "Done!"
```

Save as `delete-vpc.sh`, make executable, and run:
```bash
chmod +x delete-vpc.sh
./delete-vpc.sh vpc-xxxxx
```

## Important Notes

⚠️ **WARNING**: 
- This will permanently delete the VPC and all its resources
- Make sure you don't have any important resources in the VPC
- The default VPC cannot be deleted (but you can recreate it)
- NAT Gateways take a few minutes to delete

## After Deleting VPCs

Once you've freed up VPC slots, you can deploy:

```bash
cd terraform
terraform plan
terraform apply
```

This will create a fresh VPC for your TOVA deployment.

