output "file_system_id" {
  value = aws_efs_file_system.main.id
}

output "file_system_arn" {
  value = aws_efs_file_system.main.arn
}

output "dns_name" {
  value = aws_efs_file_system.main.dns_name
}

output "security_group_id" {
  value = aws_security_group.efs.id
}

