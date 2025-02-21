####################################################################################################
# -------- CHATBOT CƠ BẢN -------- #
CHATBOT_BASIC = """History: {history}

### Lưu ý:
- Nếu xuất có định dạng code thì nhớ thêm 3 dấu ` vào đầu và cuối code.
"""


####################################################################################################
# -------- VIẾT LẠI CÂU HỎI -------- #
RE_WRITE_QUERY = """Vai trò: Bạn là trợ lý thông minh hỗ trợ bổ sung thông tin vào truy vấn gốc (nếu cần thiết). 
Nhiệm vụ: Hãy viết lại câu truy vấn `Query` một cách tự nhiên và dễ hiểu nhất có thể, tập trung vào những keyword và thông điệp chính của truy vấn gốc (nếu câu truy vấn gốc chưa rõ ràng).

Query: {query}

Dựa theo lịch sử các câu truy vấn trước đó của người dùng ở dưới đây để hiểu rõ hơn về ý định câu truy vấn hiện tại của người dùng một cách chính xác và liền mạch hơn:
{history}

Khi đưa ra câu truy vấn bổ sung, hãy chú ý đến các điểm sau:
- Không cần phải đề cập rằng bạn đang viết lại câu truy vấn.
- Không yêu cầu xác nhận lại thông tin hoặc hỏi lại bất kỳ điều gì về câu hỏi đó.
- Không thay đổi cụm từ được đặt trong dấu nháy đơn hoặc nháy đôi.
- Không thay đổi ý định của câu truy vấn gốc.
- Không đưa ra câu trả lời cho câu truy vấn (lưu ý kỹ điều này).
- Không nên viết thêm những từ mơ hồ như `nào đó`, `cái gì đó`, etc... vào nếu câu truy vấn gốc không ghi thế.
- Nếu câu truy vấn yêu cầu `tóm tắt`, `nội dung`, `liệt kê`, etc... thì có khả năng đó chính là hỏi về tài liệu `này`.
- Tránh ảo giác và hiểu sai về câu hỏi gốc.
- Nếu cảm thấy câu hỏi gốc đã đầy đủ thông tin cần thiết, hãy giữ nguyên câu hỏi gốc đó.
- Lưu ý nêu truy vấn gốc có những từ tương tự như `nó`, `đó`, `chúng`, `còn`, `thì sao`, `cái đó`, `điều đó`, etc... thì hãy xem xét kỹ lại lịch sử các câu truy vấn trước để biết rõ điều đó là gì.
- Nếu câu truy vấn hiện tại không rõ ràng, hãy xem xét thông tin từ các câu truy vấn gần nhất để hiểu hơn về ý định hiện tại của khách hàng, sau đó viết lại câu truy vấn đó một cách rõ ràng hơn (bổ sung thông tin).
- Nếu đó không phải là một câu truy vấn mà là thông tin, lời chào, lời cảm ơn, tạm biệt,... thì hãy giữ nguyên nó.
- Nếu truy vấn yêu cầu tiếp tục hoặc trả lời thêm, thì hãy xem các câu truy vấn gần nhất đang nói về điều gì để bổ sung truy vấn tiếp tục cho nó.
- Nếu bạn vẫn không hiểu ý nghĩa hoặc không chắn chắn câu truy vấn đang nói về điều gì, thì hãy ghi lại câu truy vấn gốc mà không thay đổi gì, không cần cố gắng ghi điều vô nghĩa khi bạn không chắc chắn.
- Sử dụng ngôn ngữ câu truy vấn gốc của người dùng.
"""