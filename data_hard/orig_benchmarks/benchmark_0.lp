d6(o6).
d3(o13).
d4(o0).
d4(o5).
d11(o6).
d10(o12).
d1(o10).
d12(o10).
d8(o10).
d14(o12).
d4(o9).
d5(o1).
d1(o2).
d7(o13).
d0(o2).

d3(X) :- d10(X), not d9(X), d1(X).
d2(X) :- d0(X), not d8(X), d1(X).
d13(X) :- not d3(X), d1(X), d9(X).
d7(X) :- d5(X), not d2(X), d9(X).
d5(X) :- d7(X), not d9(X), d8(X).
d13(X) :- d10(X), d14(X), not d4(X).
d3(X) :- d10(X), d14(X), not d6(X).
d9(X) :- not d10(X), d11(X), d2(X).
d9(X) :- not d7(X), d14(X), d1(X).
d2(X) :- d10(X), d3(X), not d4(X).
d0(X) :- not d11(X), d1(X), d2(X).
d1(X) :- d14(X), not d5(X), d2(X).
d2(X) :- d5(X), not d0(X), d8(X).