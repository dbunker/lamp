d3(o12).
d11(o15).
d0(o5).
d6(o2).
d7(o0).
d12(o8).
d1(o12).
d10(o0).
d17(o4).
d17(o10).

d15(X) :- not d7(X), d10(X), not d2(X).
d5(X) :- not d17(X), d9(X), not d15(X).
d12(X) :- not d4(X), not d10(X), d15(X).
d4(X) :- not d10(X), not d6(X), d11(X).
d13(X) :- not d16(X), d12(X), not d5(X).
d16(X) :- not d13(X), d4(X), not d2(X).
d14(X) :- not d12(X), not d9(X), d3(X).
d11(X) :- not d4(X), not d15(X), d5(X).
d8(X) :- not d3(X), not d10(X), d9(X).
d9(X) :- not d0(X), d16(X), not d17(X).
d14(X) :- not d4(X), d17(X), not d16(X).
d1(X) :- not d4(X), d11(X), not d15(X).
d12(X) :- not d0(X), not d9(X), d5(X).
d8(X) :- not d17(X), not d1(X), d3(X).
d6(X) :- not d7(X), d10(X), not d2(X).
d16(X) :- not d0(X), d9(X), not d15(X).
d8(X) :- not d10(X), d14(X), not d12(X).
d2(X) :- not d4(X), d0(X), not d10(X).
d10(X) :- not d4(X), not d7(X), d9(X).
d3(X) :- not d0(X), not d11(X), d13(X).