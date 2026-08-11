<?php
/**
 * Standalone Face Registration API
 * Registers a new face descriptor for an employee if it is not already registered.
 */

// CORS & Referrer Policy headers
$origin = isset($_SERVER['HTTP_ORIGIN']) ? $_SERVER['HTTP_ORIGIN'] : '';
// if (empty($origin) && isset($_SERVER['HTTP_REFERER'])) {
//     $parsed_referer = parse_url($_SERVER['HTTP_REFERER']);
//     if (isset($parsed_referer['scheme']) && isset($parsed_referer['host'])) {
//         $origin = $parsed_referer['scheme'] . '://' . $parsed_referer['host'];
//         if (isset($parsed_referer['port'])) {
//             $origin .= ':' . $parsed_referer['port'];
//         }
//     }
// }
// if (empty($origin) || $origin === '*') {
//     $origin = 'https://calcsalary.ematrixinfotech.com';
// }

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
$env = 'local'; // Options: local, dev, prod
$db_urls = [
    'local' => 'mysql+mysqlconnector://root:@localhost/calcsalary',
    'prod'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/ematrix_calcsalary',
    'demo'  => 'mysql+mysqlconnector://admin:01eMatrix007!@69.57.172.154:3306/demo_calcsalary',
];
define('FACE_MATCH_THRESHOLD', 0.45);

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

// --- DB URL Parser ---
function parse_db_url($url) {
    $cleaned_url = str_replace('mysql+mysqlconnector://', 'mysql://', $url);
    $parsed = parse_url($cleaned_url);
    if (!$parsed) {
        throw new Exception("Failed to parse database URL.");
    }
    
    $host = isset($parsed['host']) ? $parsed['host'] : 'localhost';
    $port = isset($parsed['port']) ? $parsed['port'] : '3306';
    $user = isset($parsed['user']) ? urldecode($parsed['user']) : 'root';
    $pass = isset($parsed['pass']) ? urldecode($parsed['pass']) : '';
    $dbname = isset($parsed['path']) ? ltrim($parsed['path'], '/') : '';
    
    return [
        'host' => $host,
        'port' => $port,
        'user' => $user,
        'pass' => $pass,
        'dbname' => $dbname
    ];
}

// --- Euclidean Distance ---
function euclidean_distance($a, $b) {
    if (!is_array($a) || !is_array($b) || count($a) !== count($b)) {
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

// --- Request Reader ---
$data = $_POST;
$json = file_get_contents('php://input');
if ($json) {
    $decoded = json_decode($json, true);
    if (is_array($decoded)) {
        $data = array_merge($data, $decoded);
    }
}

$employeeId = isset($data['employeeId']) ? intval($data['employeeId']) : null;
$faceDescriptorRaw = isset($data['faceDescriptor']) ? $data['faceDescriptor'] : null;

header('Content-Type: application/json');

if ($employeeId === null || $faceDescriptorRaw === null) {
    http_response_code(400);
    echo json_encode(["detail" => "employeeId and faceDescriptor are required parameters."]);
    exit;
}

$new_descriptor = json_decode($faceDescriptorRaw, true);
if (json_last_error() !== JSON_ERROR_NONE || !is_array($new_descriptor)) {
    http_response_code(400);
    echo json_encode(["detail" => "Invalid faceDescriptor format. Must be a JSON-encoded array of numbers."]);
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

// --- Registration Business Logic ---
try {
    // Ensure employee exists
    $stmt = $db->prepare("SELECT * FROM company_employees WHERE id = :id");
    $stmt->execute(['id' => $employeeId]);
    $employee = $stmt->fetch();

    if (!$employee) {
        http_response_code(404);
        echo json_encode(["detail" => "Employee ID not found."]);
        exit;
    }

    // Check if this employee already has a registered face
    if ($employee['embedding'] !== null && trim($employee['embedding']) !== '') {
        http_response_code(400);
        echo json_encode(["detail" => "This employee already has a registered face. Please clear the existing face first."]);
        exit;
    }

    // Check if the face is already registered with another employee
    $other_stmt = $db->prepare("SELECT id, user_name, embedding FROM company_employees WHERE embedding IS NOT NULL AND id != :id");
    $other_stmt->execute(['id' => $employeeId]);
    $other_employees = $other_stmt->fetchAll();

    foreach ($other_employees as $other) {
        $saved_embedding = $other['embedding'];
        if (is_string($saved_embedding)) {
            $saved_embedding = json_decode($saved_embedding, true);
        }

        if (is_array($saved_embedding)) {
            $dist = euclidean_distance($new_descriptor, $saved_embedding);
            if ($dist < FACE_MATCH_THRESHOLD) {
                http_response_code(400);
                echo json_encode(["detail" => "This face is already registered."]);
                exit;
            }
        }
    }

    // Save new descriptor
    $db->beginTransaction();
    $update_stmt = $db->prepare("UPDATE company_employees SET embedding = :embedding WHERE id = :id");
    $update_stmt->execute([
        'embedding' => json_encode($new_descriptor),
        'id' => $employeeId
    ]);
    $db->commit();

    echo json_encode(["success" => true, "message" => "Face registered successfully."]);

} catch (Exception $e) {
    if ($db->inTransaction()) {
        $db->rollBack();
    }
    http_response_code(500);
    echo json_encode(["detail" => "Registration failed: " . $e->getMessage()]);
}
