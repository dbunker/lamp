d13(o12).
d7(o6).
d2(o2).
d6(o12).
d4(o5).
d3(o8).
d6(o0).
d10(o3).
d8(o8).
d1(o8).

d0(X) :- d2(X), d8(X), not d12(X).
d5(X) :- d8(X), d0(X), not d10(X).
d8(X) :- not d6(X), d9(X), d3(X).
d12(X) :- not d1(X), d8(X), d5(X).
d4(X) :- not d5(X), d1(X), d11(X).
d11(X) :- not d12(X), d7(X), d9(X).
d0(X) :- d13(X), d6(X), not d2(X).
d6(X) :- d10(X), d11(X), not d2(X).
d4(X) :- d2(X), d7(X), not d5(X).
d6(X) :- not d7(X), d10(X), d9(X).
d6(X) :- not d5(X), d0(X), d3(X).
d9(X) :- not d7(X), d1(X), d12(X).
d2(X) :- d4(X), d1(X), not d7(X).
d4(X) :- not d7(X), d1(X), d3(X).
d9(X) :- not d1(X), d4(X), d6(X).
d12(X) :- not d7(X), d13(X), d5(X).
d12(X) :- d1(X), d7(X), not d11(X).
d4(X) :- not d11(X), d9(X), d5(X).
d11(X) :- not d12(X), d4(X), d5(X).
d2(X) :- not d9(X), d13(X), d3(X).