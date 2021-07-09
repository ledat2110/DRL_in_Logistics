- Cài đặt thư viện
	+ chạy file setup_lib.sh để thực hiện cài đặt thư viện
	bash setup_lib.sh
	
- Thực hiện huấn luyện:
	+ phương pháp đạo hàm chính sách
	python vpg_method.py -n <tên lần huấn luyện> [-m <mô hình mạng sử dụng huấn luyện tiếp>] [-s: huấn luyện không dừng]
	+ Phương pháp đạo hàm chính sách sử dụng mô hình đề xuất
	python matrix_vpg_method.py -n <tên lần huấn luyện> [-m <mô hình mạng sử dụng huấn luyện tiếp>] [-s: huấn luyện không dừng]
	+ phương pháp đạo hàm chính sách đa chủ thể
	Huấn luyện chủ thể quản lý các đơn vị bán lẻ
	python retailer_vpg_method.py -n <tên lần huấn luyện> [-m <mô hình mạng sử dụng huấn luyện tiếp>] [-s: huấn luyện không dừng]
	Huấn luyện chủ thể quản lý nhà máy, nhà phân phối
	python warehouse_vpg_method.py -n <tên lần huấn luyện> -rm <tên model chủ thể quản lý đơn vị bán lẻ> [-m <model pretrained>] [-s: huấn luyện không dừng]

- Các mô hình lưu trong thư mục:
./saves/<loại pp>-<tên lần huấn luyện>/

- Thực hiên đánh giá các phương pháp
	+ phương pháp chính sách heuristic (zeta, Q)
	python test.py -t threshold -m None [-n <số chu kỳ chạy>] [-p: vẽ biểu đồ kết quả] [-tr kích hoạt đơn hàng có xu hướng] [-v kích hoạt đơn hàng dao động mạnh]
	+ phương pháp đạo hàm chính sách
	python test.py -t vpg -m <tên mô hình mạng> [-n <số chu kỳ chạy>] [-p: vẽ biểu đồ kết quả] [-tr kích hoạt đơn hàng có xu hướng] [-v kích hoạt đơn hàng dao động mạnh]
	+ phương pháp đạo hàm chính sách sử dụng mô hình đề xuất
	python test.py -t matrix_vpg -m <tên mô hình mạng> [-n <số chu kỳ chạy>] [-p: vẽ biểu đồ kết quả] [-tr kích hoạt đơn hàng có xu hướng] [-v kích hoạt đơn hàng dao động mạnh]
	+ phương pháp đạo hàm chính sách đa chủ thể
	python test.py -t 2_agent -m <tên mô hình mạng chủ thể quản lý nhà máy, nhà phân phối> -rm <tên mô hình mạng chủ thể quản lý đơn vị bán lẻ> [-n <số chu kỳ chạy>] [-p: vẽ biểu đồ kết quả] [-tr kích hoạt đơn hàng có xu hướng] [-v kích hoạt đơn hàng dao động mạnh]
	
- Thực hiện chạy demo phiên bản web:
	streamlit run web_demo.p
