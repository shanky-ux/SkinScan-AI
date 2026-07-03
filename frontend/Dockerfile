FROM node:20-alpine

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json* /app/
RUN npm install

COPY frontend /app

ENV NEXT_PUBLIC_API_URL=http://localhost:8000

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]