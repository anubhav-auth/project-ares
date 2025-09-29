package me.anubhav_auth.analytics_service.entity;

import jakarta.persistence.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity // Tells Spring this class is a table in the database
@Table(name = "applications") // Specifies the table name
public class Application {

    @Id // Marks this field as the Primary Key
    @GeneratedValue(strategy = GenerationType.IDENTITY) // Auto-increments the ID
    private Long id;

    @Column(name = "job_id", nullable = false)
    private String jobId;

    @Column(name = "job_title", nullable = false)
    private String jobTitle;

    @Column(name = "company_name", nullable = false)
    private String companyName;

    @Column(name = "source")
    private String source;

    @Column(name = "status", nullable = false)
    private String status = "NOTIFIED"; // Sets the default value

    @Column(name = "action_type")
    private String actionType;

    @Column(name = "match_score")
    private Integer matchScore;

    @JdbcTypeCode(SqlTypes.JSON) // Handles the JSONB type for the dossier
    @Column(name = "dossier", columnDefinition = "jsonb")
    private String dossier;

    @Column(name = "generated_cover_letter", columnDefinition = "TEXT")
    private String generatedCoverLetter;

    @Column(name = "confirmation_screenshot_url")
    private String confirmationScreenshotUrl;

    @CreationTimestamp // Automatically sets the creation timestamp
    @Column(name = "created_at", updatable = false)
    private Instant createdAt;

    @UpdateTimestamp // Automatically updates the timestamp on edit
    @Column(name = "updated_at")
    private Instant updatedAt;

    // Getters and Setters - Required for the framework to work
    // You can auto-generate these in your IDE
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getJobId() { return jobId; }
    public void setJobId(String jobId) { this.jobId = jobId; }
    public String getJobTitle() { return jobTitle; }
    public void setJobTitle(String jobTitle) { this.jobTitle = jobTitle; }
    public String getCompanyName() { return companyName; }
    public void setCompanyName(String companyName) { this.companyName = companyName; }
    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public String getActionType() { return actionType; }
    public void setActionType(String actionType) { this.actionType = actionType; }
    public Integer getMatchScore() { return matchScore; }
    public void setMatchScore(Integer matchScore) { this.matchScore = matchScore; }
    public String getDossier() { return dossier; }
    public void setDossier(String dossier) { this.dossier = dossier; }
    public String getGeneratedCoverLetter() { return generatedCoverLetter; }
    public void setGeneratedCoverLetter(String generatedCoverLetter) { this.generatedCoverLetter = generatedCoverLetter; }
    public String getConfirmationScreenshotUrl() { return confirmationScreenshotUrl; }
    public void setConfirmationScreenshotUrl(String confirmationScreenshotUrl) { this.confirmationScreenshotUrl = confirmationScreenshotUrl; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(Instant updatedAt) { this.updatedAt = updatedAt; }
}