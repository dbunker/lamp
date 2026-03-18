d5(o3).
d3(o10).
d5(o0).
d12(o9).
d14(o1).
d8(o2).
d9(o10).
d4(o5).
d9(o6).
d17(o3).
d4(o2).
d10(o3).
d6(o13).
d7(o5).
d17(o10).

d0(X) :- not d7(X), d9(X), not d12(X).
d1(X) :- d17(X), not d9(X), not d15(X).
d5(X) :- d12(X), not d14(X), not d2(X).
d5(X) :- d2(X), not d16(X), not d14(X).
d11(X) :- not d1(X), not d14(X), d13(X).
d7(X) :- not d1(X), not d12(X), d15(X).
d4(X) :- not d0(X), d1(X), not d5(X).
d2(X) :- not d1(X), not d11(X), d15(X).
d11(X) :- not d7(X), not d13(X), d3(X).
d7(X) :- d10(X), not d15(X), not d9(X).
d8(X) :- not d12(X), not d6(X), d11(X).
d9(X) :- d17(X), not d5(X), not d2(X).
d16(X) :- not d8(X), not d6(X), d9(X).
d2(X) :- d12(X), not d6(X), not d5(X).
d7(X) :- not d13(X), not d9(X), d11(X).
d13(X) :- not d10(X), not d14(X), d9(X).
d15(X) :- not d1(X), d11(X), not d2(X).
d15(X) :- not d7(X), not d10(X), d13(X).
d17(X) :- not d11(X), d15(X), not d2(X).
d4(X) :- not d7(X), not d3(X), d15(X).