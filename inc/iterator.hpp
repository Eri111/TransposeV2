#pragma once
#include <cmath> // abs
#include <cstdint>
#include <iterator>

namespace pad {
//change name to Arith_Iterator
template <typename F>
class Arith_Iterator {
   private:
	ssize_t idx_;
	F fun_;
   public:
	using iterator_category = std::forward_iterator_tag;
	using value_type = decltype(typename std::remove_reference<F>::type{}(0));
	using pointer = std::nullptr_t;
	using reference = value_type;
	using difference_type = ssize_t;

	inline Arith_Iterator() = delete;

	inline explicit Arith_Iterator(const ssize_t _idx, F&& _fun)
	    : idx_{_idx}, fun_{std::forward<F>(_fun)} {
	}

	inline Arith_Iterator(
	    const Arith_Iterator& other) = default;

	inline Arith_Iterator& operator=(
	    const Arith_Iterator& other) = default;

	// Pointer like operators
	[[nodiscard]] inline reference operator*() {
		return fun_(idx_);
	}

	[[nodiscard]] inline reference operator->() {
		return this->operator*();
	}

	// Increment / Decrement
	inline Arith_Iterator& operator++() noexcept {
		++idx_;
		return *this;
	}

	inline Arith_Iterator& operator++(int) noexcept {
		auto tmp = *this;
		++*this;
		return tmp;
	}
	
	inline Arith_Iterator& operator--() noexcept {
		--idx_;
		return *this;
	}

	inline Arith_Iterator& operator--(int) noexcept {
		auto tmp = *this;
		--*this;
		return tmp;
	}

	inline Arith_Iterator& operator+=(const ssize_t inc) noexcept {
		idx_ += inc;
		return *this;
	}

	// Arithmetic operators
	[[nodiscard]] inline friend difference_type operator-(
	    const Arith_Iterator& lhs,
	    const Arith_Iterator& rhs) noexcept {
		return std::abs(lhs.idx_ - rhs.idx_);
	}

	[[nodiscard]] inline friend Arith_Iterator operator+(
	    const Arith_Iterator& lhs,
	    const ssize_t idx) noexcept {
		auto tmp = lhs;
		tmp.idx_ += idx;
		return tmp;
	}

	// Comparision operators
	[[nodiscard]] inline friend bool operator==(
	    const Arith_Iterator& lhs,
	    const Arith_Iterator& rhs) noexcept {
		return lhs.idx_ == rhs.idx_;
	}

	[[nodiscard]] inline friend bool operator!=(
	    const Arith_Iterator& lhs,
	    const Arith_Iterator& rhs) noexcept {
		return !(lhs == rhs);
	}
};
} // namespace pad
