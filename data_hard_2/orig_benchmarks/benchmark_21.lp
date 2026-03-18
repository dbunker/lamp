d6(o6).
d13(o16).
d1(o5).
d8(o0).
d9(o11).
d10(o1).
d17(o15).
d6(o3).
d17(o2).
d15(o15).
d10(o12).
d11(o13).
d6(o14).
d12(o15).
d13(o6).

d0(X) :- d9(X), not d16(X).
d5(X) :- not d11(X), d12(X).
d14(X) :- not d0(X), d10(X).
d0(X) :- d9(X), not d16(X).
d3(X) :- not d8(X), d13(X).
d3(X) :- d9(X), not d2(X).
d2(X) :- d4(X), not d13(X).
d16(X) :- d7(X), not d3(X).
d5(X) :- d1(X), not d10(X).
d7(X) :- d0(X), not d15(X).
d0(X) :- not d10(X), d11(X).
d0(X) :- d2(X), not d5(X).
d8(X) :- d0(X), not d6(X).
d1(X) :- not d6(X), d3(X).
d4(X) :- not d8(X), d13(X).