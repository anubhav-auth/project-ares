package me.anubhav_auth.analytics_service.controller;

import lombok.RequiredArgsConstructor;
import me.anubhav_auth.analytics_service.repository.ApplicationRepository;
import me.anubhav_auth.analytics_service.repository.ErrorRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/health")
@RequiredArgsConstructor
public class HealthController {

    private final ApplicationRepository applicationRepository;
    private final ErrorRepository errorRepository;

    @GetMapping
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> health = new HashMap<>();
        health.put("status", "UP");
        health.put("timestamp", Instant.now());
        health.put("totalApplications", applicationRepository.count());
        health.put("totalErrors", errorRepository.count());

        return ResponseEntity.ok(health);
    }
}