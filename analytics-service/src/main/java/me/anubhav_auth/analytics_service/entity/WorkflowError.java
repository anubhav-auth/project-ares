package me.anubhav_auth.analytics_service.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.CreationTimestamp;

import java.time.Instant;

@Entity
@Data
@Table(name = "workflow_errors",
        indexes = {
                @Index(name = "idx_job_id_error", columnList = "job_id"),
                @Index(name = "idx_occurred_at", columnList = "occurred_at")
        })
public class WorkflowError {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "status_code")
    private Integer statusCode;

    @Column(name = "job_id")
    private String jobId;

    @Column(name = "node_name")
    private String nodeName;

    @Column(name = "workflow_name")
    private String workflowName;

    @Column(name = "execution_id")
    private String executionId;

    @Column(name = "error_type")
    private String errorType; // API_ERROR, VALIDATION_ERROR, TIMEOUT, etc.

    @Column(name = "stack_trace", columnDefinition = "TEXT")
    private String stackTrace;

    @Column(name = "retry_count")
    private Integer retryCount = 0;

    @Column(name = "resolved")
    private Boolean resolved = false;

    @CreationTimestamp
    @Column(name = "occurred_at")
    private Instant occurredAt;
}