d11(o12).
d8(o5).
d9(o2).
d7(o0).
d6(o12).
d0(o10).
d5(o5).
d4(o5).
d6(o6).
d2(o9).
d6(o0).
d3(o7).
d12(o17).
d13(o10).
d10(o4).

d1(X) :- d0(X), d12(X), not d9(X).
d12(X) :- not d4(X), d10(X), d6(X).
d5(X) :- not d13(X), d9(X), d11(X).
d12(X) :- not d0(X), d1(X), d5(X).
d10(X) :- d2(X), d7(X), not d6(X).
d12(X) :- d2(X), not d13(X), d8(X).
d11(X) :- d0(X), d13(X), not d12(X).
d2(X) :- d0(X), not d6(X), d11(X).
d9(X) :- not d8(X), d6(X), d11(X).
d4(X) :- d10(X), d7(X), not d2(X).
d12(X) :- not d5(X), d10(X), d6(X).
d8(X) :- d4(X), d9(X), not d5(X).
d7(X) :- d8(X), not d3(X), d5(X).
d4(X) :- d2(X), d12(X), not d11(X).
d12(X) :- not d10(X), d7(X), d11(X).
d6(X) :- d8(X), not d9(X), d1(X).
d3(X) :- not d7(X), d0(X), d5(X).
d8(X) :- d2(X), d0(X), not d11(X).
d2(X) :- not d8(X), d3(X), d11(X).
d5(X) :- d1(X), d12(X), not d3(X).