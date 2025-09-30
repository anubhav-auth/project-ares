package me.anubhav_auth.analytics_service.controller;

import lombok.RequiredArgsConstructor;
import me.anubhav_auth.analytics_service.entity.WorkflowError;
import me.anubhav_auth.analytics_service.repository.ErrorRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;

@RestController
@RequestMapping("/api/errors")
@RequiredArgsConstructor
public class ErrorController {

    private final ErrorRepository errorRepository;

    @PostMapping
    public ResponseEntity<WorkflowError> logError(@RequestBody WorkflowError error) {
        return ResponseEntity.ok(errorRepository.save(error));
    }

    @GetMapping
    public ResponseEntity<List<WorkflowError>> getAllErrors() {
        return ResponseEntity.ok(errorRepository.findAll());
    }

    @GetMapping("/recent")
    public ResponseEntity<List<WorkflowError>> getRecentErrors(
            @RequestParam(defaultValue = "24") int hours) {
        Instant since = Instant.now().minus(hours, ChronoUnit.HOURS);
        return ResponseEntity.ok(errorRepository.findByOccurredAtAfterOrderByOccurredAtDesc(since));
    }

    @GetMapping("/by-job/{jobId}")
    public ResponseEntity<List<WorkflowError>> getErrorsByJobId(@PathVariable String jobId) {
        return ResponseEntity.ok(errorRepository.findByJobId(jobId));
    }

    @DeleteMapping("/older-than/{days}")
    public ResponseEntity<Void> deleteOldErrors(@PathVariable int days) {
        Instant cutoff = Instant.now().minus(days, ChronoUnit.DAYS);
        errorRepository.deleteByOccurredAtBefore(cutoff);
        return ResponseEntity.noContent().build();
    }
}