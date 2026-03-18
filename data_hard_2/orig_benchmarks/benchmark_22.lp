d7(o12).
d15(o4).
d6(o2).
d5(o1).
d9(o2).
d0(o2).
d16(o7).
d2(o12).
d16(o3).
d10(o7).
d13(o16).
d2(o7).
d8(o12).
d4(o6).
d3(o14).

d14(X) :- d12(X), d11(X), not d2(X).
d14(X) :- not d7(X), d17(X), d15(X).
d2(X) :- not d4(X), d13(X), d3(X).
d13(X) :- not d12(X), d7(X), d15(X).
d2(X) :- not d8(X), d7(X), d6(X).
d6(X) :- not d13(X), d8(X), d7(X).
d1(X) :- d17(X), d13(X), not d16(X).
d9(X) :- d2(X), not d14(X), d13(X).
d8(X) :- d2(X), not d0(X), d13(X).
d13(X) :- not d12(X), d11(X), d15(X).
d11(X) :- not d8(X), d10(X), d14(X).
d7(X) :- d2(X), not d4(X), d13(X).
d12(X) :- d10(X), d14(X), not d9(X).
d15(X) :- not d4(X), d8(X), d7(X).
d17(X) :- d0(X), not d14(X), d9(X).
d3(X) :- not d1(X), d17(X), d9(X).
d14(X) :- d1(X), d17(X), not d10(X).
d11(X) :- d4(X), d16(X), not d9(X).
d13(X) :- not d4(X), d10(X), d3(X).
d3(X) :- not d4(X), d0(X), d15(X).