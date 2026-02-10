output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.tova.id
}

output "public_ip" {
  description = "Public IP address (Elastic IP)"
  value       = aws_eip.tova.public_ip
}

output "public_dns" {
  description = "Public DNS name"
  value       = aws_instance.tova.public_dns
}

output "ssh_command" {
  description = "SSH command to connect to instance (uses ubuntu for Ubuntu, ec2-user for Amazon Linux)"
  value       = "ssh -i your-key.pem ubuntu@${aws_eip.tova.public_ip}"
}

output "web_url" {
  description = "Web UI URL"
  value       = "http://${aws_eip.tova.public_ip}:8080"
}

output "api_url" {
  description = "API URL"
  value       = "http://${aws_eip.tova.public_ip}:8000"
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.ec2.id
}

output "instance_arn" {
  description = "EC2 instance ARN"
  value       = aws_instance.tova.arn
}

