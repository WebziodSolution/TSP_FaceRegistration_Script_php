<?php
/**
 * Standalone Face Recognition Login API
 * Matches uploaded face image / InsightFace 512D ArcFace embedding against database embeddings.
 */

// CORS & Referrer Policy headers
$origin = isset($_SERVER['HTTP_ORIGIN']) ? $_SERVER['HTTP_ORIGIN'] : '';
if (!empty($origin)) {
    header("Access-Control-Allow-Origin: " . $origin);
} else {
    header("Access-Control-Allow-Origin: *");
}
header("Access-Control-Allow-Credentials: true");
header("Access-Control-Allow-Methods: POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization, X-Requested-With");
header("Referrer-Policy: strict-origin-when-cross-origin");

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

// --- Configuration ---
$env = 'demo'; // Options: local, dev, prod
$db_urls = [
    'local' => 'mysql+mysqlconnector://root:@localhost/calcsalary',
    'prod'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/ematrix_calcsalary',
    'demo'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/demo_calcsalary',
];

define('INSIGHTFACE_SERVICE_URL', 'https://presentapi.ematrixinfotech.com/py/extract');
// InsightFace ArcFace (buffalo_s) Cosine Similarity Threshold:
// Normal matching: 0.45 - 0.50 | Strict matching: 0.55 - 0.60
define('COSINE_SIMILARITY_THRESHOLD', 0.50); 
define('EUCLIDEAN_MATCH_THRESHOLD', 0.45);   // Fallback for legacy 128D descriptors

// IP restriction: apply only in production
$client_ip = isset($_SERVER['REMOTE_ADDR']) ? $_SERVER['REMOTE_ADDR'] : '';
if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
    $ips = explode(',', $_SERVER['HTTP_X_FORWARDED_FOR']);
    $client_ip = trim($ips[0]);
} elseif (!empty($_SERVER['HTTP_CLIENT_IP'])) {
    $client_ip = $_SERVER['HTTP_CLIENT_IP'];
}

$is_allowed = true;
if ($env === 'prod') {
    $is_allowed = false;
    if (strpos($client_ip, '192.168.') === 0 || $client_ip === '127.0.0.1' || $client_ip === '150.129.166.66' || $client_ip === '::1') {
        $is_allowed = true;
    }
}

if (!$is_allowed) {
    http_response_code(403);
    header('Content-Type: application/json');
    echo json_encode(["detail" => "Forbidden: Access denied from IP " . $client_ip]);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    header('Content-Type: application/json');
    echo json_encode(["detail" => "Method Not Allowed. Use POST."]);
    exit;
}

// --- Helper: DB URL Parser ---
function parse_db_url($url) {
    $cleaned_url = str_replace('mysql+mysqlconnector://', 'mysql://', $url);
    $parsed = parse_url($cleaned_url);
    if (!$parsed) {
        throw new Exception("Failed to parse database URL.");
    }
    
    return [
        'host'   => isset($parsed['host']) ? $parsed['host'] : 'localhost',
        'port'   => isset($parsed['port']) ? $parsed['port'] : '3306',
        'user'   => isset($parsed['user']) ? urldecode($parsed['user']) : 'root',
        'pass'   => isset($parsed['pass']) ? urldecode($parsed['pass']) : '',
        'dbname' => isset($parsed['path']) ? ltrim($parsed['path'], '/') : ''
    ];
}

// --- Helper: Cosine Similarity for InsightFace 512D Vectors ---
function cosine_similarity(array $a, array $b) {
    $count = count($a);
    if ($count !== count($b) || $count === 0) {
        return -1.0;
    }
    $dot = 0.0;
    $normA = 0.0;
    $normB = 0.0;
    for ($i = 0; $i < $count; $i++) {
        $dot   += $a[$i] * $b[$i];
        $normA += $a[$i] * $a[$i];
        $normB += $b[$i] * $b[$i];
    }
    if ($normA <= 0 || $normB <= 0) {
        return -1.0;
    }
    return $dot / (sqrt($normA) * sqrt($normB));
}

// --- Helper: Euclidean Distance for Legacy 128D Vectors ---
function euclidean_distance(array $a, array $b) {
    if (count($a) !== count($b)) {
        return 999.0;
    }
    $sum = 0.0;
    $count = count($a);
    for ($i = 0; $i < $count; $i++) {
        $diff = $a[$i] - $b[$i];
        $sum += $diff * $diff;
    }
    return sqrt($sum);
}

// --- Helper: Call InsightFace Microservice ---
function extract_embedding_from_insightface($filePath = null, $base64Data = null) {
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, INSIGHTFACE_SERVICE_URL);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 15);

    if ($filePath && file_exists($filePath)) {
        $mime = mime_content_type($filePath) ?: 'image/jpeg';
        $cfile = new CURLFile($filePath, $mime, basename($filePath));
        curl_setopt($ch, CURLOPT_POSTFIELDS, ['file' => $cfile]);
    } elseif ($base64Data) {
        curl_setopt($ch, CURLOPT_POSTFIELDS, ['image_base64' => $base64Data]);
    } else {
        throw new Exception("No image data provided for face extraction.");
    }

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlError = curl_error($ch);
    curl_close($ch);

    if ($curlError) {
        throw new Exception("InsightFace service connection error: " . $curlError . ". Make sure the Python service is running on port 8000.");
    }

    $result = json_decode($response, true);
    if (!$result || !isset($result['success'])) {
        throw new Exception("Invalid response from InsightFace service: " . $response);
    }

    if ($result['success'] !== true) {
        throw new Exception($result['detail'] ?? "Face extraction failed.");
    }

    return $result['embedding'];
}

// --- Request Reader ---
$data = $_POST;
$json = file_get_contents('php://input');
if ($json) {
    $decoded = json_decode($json, true);
    if (is_array($decoded)) {
        $data = array_merge($data, $decoded);
    }
}

header('Content-Type: application/json');

$login_descriptor = null;

// Case 1: Image file uploaded via FormData
if (isset($_FILES['image']) && $_FILES['image']['error'] === UPLOAD_ERR_OK) {
    try {
        $login_descriptor = extract_embedding_from_insightface($_FILES['image']['tmp_name'], null);
    } catch (Exception $e) {
        http_response_code(400);
        echo json_encode(["detail" => $e->getMessage()]);
        exit;
    }
}
// Case 2: Base64 image data uploaded
elseif (isset($data['image']) && is_string($data['image']) && strpos($data['image'], 'data:image') === 0) {
    try {
        $login_descriptor = extract_embedding_from_insightface(null, $data['image']);
    } catch (Exception $e) {
        http_response_code(400);
        echo json_encode(["detail" => $e->getMessage()]);
        exit;
    }
}
// Case 3: Pre-computed descriptor passed in
elseif (isset($data['faceDescriptor'])) {
    $raw = $data['faceDescriptor'];
    if (is_string($raw)) {
        $raw = json_decode($raw, true);
    }
    if (is_array($raw)) {
        $login_descriptor = $raw;
    }
}

if (!$login_descriptor || !is_array($login_descriptor)) {
    http_response_code(400);
    echo json_encode(["detail" => "No valid face image or descriptor provided for login."]);
    exit;
}

// --- Database Connection ---
$db_url = isset($db_urls[$env]) ? $db_urls[$env] : null;
if (!$db_url) {
    http_response_code(500);
    echo json_encode(["detail" => "Database URL configuration for environment '$env' not found."]);
    exit;
}

try {
    $db_config = parse_db_url($db_url);
    $dsn = "mysql:host={$db_config['host']};port={$db_config['port']};dbname={$db_config['dbname']};charset=utf8mb4";
    $options = [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES   => false,
    ];
    $db = new PDO($dsn, $db_config['user'], $db_config['pass'], $options);
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(["detail" => "Database connection failed: " . $e->getMessage()]);
    exit;
}

// --- Login Business Logic ---
try {
    // Retrieve all employees with registered faces
    $stmt = $db->query("SELECT id, company_id, user_name, password, embedding FROM company_employees WHERE embedding IS NOT NULL");
    $employees = $stmt->fetchAll();

    $loginDim = count($login_descriptor);
    $best_match = null;
    $highest_similarity = -1.0;
    $best_distance = 999.0;

    foreach ($employees as $employee) {
        $saved_embedding = $employee['embedding'];
        if (is_string($saved_embedding)) {
            $saved_embedding = json_decode($saved_embedding, true);
        }

        if (is_array($saved_embedding)) {
            $savedDim = count($saved_embedding);

            // InsightFace 512D ArcFace Cosine Matching
            if ($loginDim === 512 && $savedDim === 512) {
                $sim = cosine_similarity($login_descriptor, $saved_embedding);
                if ($sim > $highest_similarity) {
                    $highest_similarity = $sim;
                    if ($sim >= COSINE_SIMILARITY_THRESHOLD) {
                        $best_match = $employee;
                    }
                }
            }
            // Legacy 128D Euclidean Matching
            elseif ($loginDim === 128 && $savedDim === 128) {
                $dist = euclidean_distance($login_descriptor, $saved_embedding);
                if ($dist < $best_distance) {
                    $best_distance = $dist;
                    if ($dist < EUCLIDEAN_MATCH_THRESHOLD) {
                        $best_match = $employee;
                    }
                }
            }
        }
    }

    if ($best_match !== null) {
        $company_details = null;
        if ($best_match['company_id'] !== null) {
            $company_stmt = $db->prepare("SELECT company_no FROM company_details WHERE id = :company_id");
            $company_stmt->execute(['company_id' => $best_match['company_id']]);
            $company_details = $company_stmt->fetch();
        }

        $confidence = $loginDim === 512 
            ? round($highest_similarity * 100, 2) . "%" 
            : round((1.0 - ($best_distance / 2.0)) * 100, 2) . "%";

        echo json_encode([
            "success" => true,
            "userName" => $best_match['user_name'],
            "companyId" => $company_details ? $company_details['company_no'] : null,
            "password" => $best_match['password'],
            "similarity" => $loginDim === 512 ? floatval($highest_similarity) : null,
            "distance" => $loginDim === 128 ? floatval($best_distance) : null,
            "confidence" => $confidence,
            "message" => "Login successful!"
        ]);
    } else {
        http_response_code(401);
        echo json_encode(["detail" => "Face not recognized. Please login using Username & Password."]);
    }

} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(["detail" => "Login failed: " . $e->getMessage()]);
}
