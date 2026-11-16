"""سرور MCP: ابزارهای عمومی دستیار را برای کلاینت‌هایی مثل Claude Desktop در دسترس می‌گذارد.
اجرا:  python mcp_server.py   (ارتباط از طریق stdio)"""
from mcp.server.mcpserver import MCPServer

from tools import calculate_usage, get_product, search_products, search_regulations

# citizen_service عمداً اینجا نیست: اطلاعات شخصی فقط بعد از احراز هویت با تصویر در خود برنامه داده می‌شود
PUBLIC_TOOLS = [search_products, get_product, calculate_usage, search_regulations]

mcp = MCPServer(
    "sabzdasht",
    instructions="Agricultural product catalog, dose/cost calculator and company regulations of Sabzdasht. "
                 "Data is in Persian; prices are in Toman.",
)
for tool in PUBLIC_TOOLS:
    mcp.tool()(tool.func)

if __name__ == "__main__":
    mcp.run()
