# EC2 Key Pair Management in Terraform

Terraform can automatically create an EC2 Key Pair for you, or you can use an existing one.

## Option 1: Create Key Pair Automatically (Recommended)

Terraform will generate a new RSA 4096-bit key pair and save the private key to a file.

### Configuration

In `terraform.tfvars`:

```hcl
deployment_type = "ec2"
ec2_create_key_pair = true  # Terraform will create the key pair
ec2_key_pair_name = ""  # Not needed when creating automatically
```

### What Happens

1. Terraform generates a new RSA 4096-bit private key
2. Creates the key pair in AWS with name: `{project_name}-{environment}-key`
3. Saves the private key to: `terraform/{project_name}-{environment}-key.pem`
4. Sets file permissions to 0400 (read-only for owner)

### After Deployment

```bash
# Get the private key path
terraform output ec2_private_key_path

# Use it to SSH
ssh -i terraform/tova-dev-key.pem ec2-user@<public_ip>
```

### Security Notes

⚠️ **Important**: 
- The private key file is saved in the `terraform/` directory
- It's automatically added to `.gitignore` to prevent committing
- Keep this file secure and never commit it to version control
- The private key is only generated once - if you lose it, you'll need to create a new key pair

## Option 2: Use Existing Key Pair

If you already have a key pair in AWS, you can use it instead.

### Configuration

In `terraform.tfvars`:

```hcl
deployment_type = "ec2"
ec2_create_key_pair = false  # Use existing key pair
ec2_key_pair_name = "my-existing-key-pair"  # Name of existing key pair in AWS
```

### Prerequisites

1. Key pair must already exist in AWS Console
2. You must have the private key file (`.pem`) on your local machine
3. Key pair name must match exactly

### Using Your Existing Key

```bash
# Use your existing private key
ssh -i ~/.ssh/my-existing-key.pem ec2-user@<public_ip>
```

## Key Pair Outputs

After deployment, Terraform provides:

```bash
# Key pair name (created or existing)
terraform output ec2_key_pair_name

# Private key file path (only if created by Terraform)
terraform output ec2_private_key_path

# Instructions for using the key
terraform output ec2_private_key_save_instructions
```

## Best Practices

1. **For Development/Testing**: Use `ec2_create_key_pair = true` (automatic)
2. **For Production**: Consider using existing key pairs managed separately
3. **Key Rotation**: Regularly rotate keys for security
4. **Backup**: Keep backups of private keys in secure storage (password manager, encrypted storage)

## Troubleshooting

### Key Pair Already Exists Error

If you get an error that the key pair already exists:

```bash
# Option 1: Use the existing key pair
ec2_create_key_pair = false
ec2_key_pair_name = "tova-dev-key"

# Option 2: Delete the existing key pair in AWS Console first
# Then run terraform apply again
```

### Can't SSH After Deployment

1. Check the private key file exists:
   ```bash
   ls -la terraform/*.pem
   ```

2. Check file permissions:
   ```bash
   chmod 400 terraform/tova-dev-key.pem
   ```

3. Verify key pair name matches:
   ```bash
   terraform output ec2_key_pair_name
   ```

4. Check security group allows SSH from your IP

### Lost Private Key

If you lose the private key file:

1. **If created by Terraform**: You'll need to create a new key pair
   ```bash
   # Update terraform.tfvars
   ec2_create_key_pair = true
   
   # Apply changes
   terraform apply
   ```

2. **If using existing key**: You'll need to use your backup or create a new key pair in AWS Console

## Example: Complete Configuration

```hcl
# terraform.tfvars
deployment_type = "ec2"

# Automatic key pair creation
ec2_create_key_pair = true
ec2_key_pair_name = ""  # Not needed

# Or use existing key pair
# ec2_create_key_pair = false
# ec2_key_pair_name = "my-existing-key"
```

After deployment:

```bash
# Get connection info
PUBLIC_IP=$(terraform output -raw ec2_public_ip)
KEY_FILE=$(terraform output -raw ec2_private_key_path)

# SSH into instance
ssh -i $KEY_FILE ec2-user@$PUBLIC_IP
```

