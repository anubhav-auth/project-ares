package me.anubhav_auth.analytics_service.entity;

import jakarta.persistence.*;
import lombok.Data;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Data
@Table(name = "applications",
        indexes = {
                @Index(name = "idx_job_id", columnList = "job_id"),
                @Index(name = "idx_status", columnList = "status"),
                @Index(name = "idx_company_name", columnList = "company_name"),
                @Index(name = "idx_created_at", columnList = "created_at")
        })
public class Application {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "job_id", nullable = false, unique = true)
    private String jobId;

    @Column(name = "job_title", nullable = false)
    private String jobTitle;

    @Column(name = "company_name", nullable = false)
    private String companyName;

    @Column(name = "source")
    private String source;

    @Column(name = "status", nullable = false)
    private String status = "NOTIFIED";

    @Column(name = "action_type")
    private String actionType;

    @Column(name = "match_score")
    private Integer matchScore;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "dossier", columnDefinition = "jsonb")
    private String dossier;

    @Column(name = "generated_cover_letter", columnDefinition = "TEXT")
    private String generatedCoverLetter;

    @Column(name = "confirmation_screenshot_url")
    private String confirmationScreenshotUrl;

    // New fields for better tracking
    @Column(name = "application_url")
    private String applicationUrl;

    @Column(name = "salary_range")
    private String salaryRange;

    @Column(name = "location")
    private String location;

    @Column(name = "job_type")
    private String jobType; // FULL_TIME, PART_TIME, CONTRACT, etc.

    @Column(name = "experience_required")
    private String experienceRequired;

    @Column(name = "skills", columnDefinition = "TEXT")
    private String skills;

    @Column(name = "notes", columnDefinition = "TEXT")
    private String notes;

    @CreationTimestamp
    @Column(name = "created_at", updatable = false)
    private Instant createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private Instant updatedAt;
}