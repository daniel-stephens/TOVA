# How to Release Elastic IPs

You've reached the Elastic IP limit (default is 5 per region). You currently have 8 EIPs allocated.

## Check Your EIPs

```bash
aws ec2 describe-addresses --query 'Addresses[*].[PublicIp,AllocationId,AssociationId,NetworkInterfaceId]' --output table
```

## Release Unused EIPs

Elastic IPs that are **not associated** with any resource can be released:

```bash
# List unassociated EIPs
aws ec2 describe-addresses --filters "Name=domain,Values=vpc" --query 'Addresses[?AssociationId==null].[AllocationId,PublicIp]' --output table

# Release an unassociated EIP
aws ec2 release-address --allocation-id eipalloc-xxxxx
```

## Release EIPs from NAT Gateways

If you have NAT Gateways you no longer need:

```bash
# List NAT Gateways
aws ec2 describe-nat-gateways --query 'NatGateways[*].[NatGatewayId,VpcId,SubnetId,State]' --output table

# Delete NAT Gateway (this releases the EIP)
aws ec2 delete-nat-gateway --nat-gateway-id nat-xxxxx

# Wait for deletion, then release the EIP
aws ec2 release-address --allocation-id eipalloc-xxxxx
```

## Release EIPs from EC2 Instances

If you have EIPs attached to EC2 instances:

```bash
# Disassociate EIP from instance
aws ec2 disassociate-address --association-id eipassoc-xxxxx

# Release the EIP
aws ec2 release-address --allocation-id eipalloc-xxxxx
```

## Quick Script to Release Unassociated EIPs

```bash
#!/bin/bash
# Release all unassociated Elastic IPs

echo "Finding unassociated Elastic IPs..."
UNASSOCIATED=$(aws ec2 describe-addresses \
  --filters "Name=domain,Values=vpc" \
  --query 'Addresses[?AssociationId==null].AllocationId' \
  --output text)

if [ -z "$UNASSOCIATED" ]; then
  echo "No unassociated EIPs found."
  exit 0
fi

echo "Found unassociated EIPs:"
for EIP in $UNASSOCIATED; do
  IP=$(aws ec2 describe-addresses --allocation-ids $EIP --query 'Addresses[0].PublicIp' --output text)
  echo "  - $EIP ($IP)"
done

read -p "Release these EIPs? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  for EIP in $UNASSOCIATED; do
    echo "Releasing $EIP..."
    aws ec2 release-address --allocation-id $EIP
  done
  echo "Done!"
else
  echo "Cancelled."
fi
```

## Request EIP Limit Increase

If you need more EIPs:

1. Go to AWS Support Center
2. Create a service limit increase request
3. Request increase for "EC2-VPC Elastic IPs"
4. Default limit is 5, you can request up to 50

## After Releasing EIPs

Once you've freed up at least 1 EIP slot, you can deploy:

```bash
cd terraform
terraform apply
```

The configuration now uses only 1 EIP (for a single NAT gateway), so you only need 1 free slot.

